"""成员 C：隐式反馈 ALS、热门度融合与最终推荐。

从项目根目录运行，例如：
  python src/implicit_als.py validate --factors 32 --iterations 10
  python src/implicit_als.py submit --factors 32 --iterations 10 --pop-weight 0

仅依赖项目当前可用的 numpy、pandas、scipy、numba。ALS 使用二值偏好
p_ui=1 和置信度 c_ui=1+alpha；未观察到的交互保留低置信度负例。
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.sparse import csr_matrix

from evaluate import evaluate, print_result


@njit(parallel=True)
def _cg_update(indptr, indices, fixed, moving, gram, alpha, regularization, steps):
    """用共轭梯度近似求解所有用户（或物品）的 ALS 正规方程。"""
    n_factors = fixed.shape[1]
    for row in prange(moving.shape[0]):
        x = moving[row]
        b = np.zeros(n_factors, dtype=np.float32)
        ax = np.zeros(n_factors, dtype=np.float32)
        for j in range(n_factors):
            ax[j] = regularization * x[j]
            for k in range(n_factors):
                ax[j] += gram[j, k] * x[k]
        for pos in range(indptr[row], indptr[row + 1]):
            y = fixed[indices[pos]]
            dot = np.float32(0.0)
            for k in range(n_factors):
                b[k] += (1.0 + alpha) * y[k]
                dot += y[k] * x[k]
            for k in range(n_factors):
                ax[k] += alpha * dot * y[k]

        residual = b - ax
        direction = residual.copy()
        old_norm = np.dot(residual, residual)
        for _ in range(steps):
            if old_norm < 1e-10:
                break
            ad = np.zeros(n_factors, dtype=np.float32)
            for j in range(n_factors):
                ad[j] = regularization * direction[j]
                for k in range(n_factors):
                    ad[j] += gram[j, k] * direction[k]
            for pos in range(indptr[row], indptr[row + 1]):
                y = fixed[indices[pos]]
                dot = np.float32(0.0)
                for k in range(n_factors):
                    dot += y[k] * direction[k]
                for k in range(n_factors):
                    ad[k] += alpha * dot * y[k]
            denominator = np.dot(direction, ad)
            if denominator <= 0:
                break
            step = old_norm / denominator
            for k in range(n_factors):
                x[k] += step * direction[k]
                residual[k] -= step * ad[k]
            new_norm = np.dot(residual, residual)
            beta = new_norm / old_norm
            for k in range(n_factors):
                direction[k] = residual[k] + beta * direction[k]
            old_norm = new_norm


class ImplicitALS:
    def __init__(self, factors=32, iterations=10, alpha=40.0,
                 regularization=0.1, cg_steps=3, seed=42):
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.regularization = regularization
        self.cg_steps = cg_steps
        self.seed = seed

    def fit(self, interactions):
        self.user_ids = np.unique(interactions['user_id'].to_numpy(dtype=np.int32))
        self.item_ids = np.unique(interactions['item_id'].to_numpy(dtype=np.int32))
        users = np.searchsorted(self.user_ids, interactions['user_id'].to_numpy(dtype=np.int32))
        items = np.searchsorted(self.item_ids, interactions['item_id'].to_numpy(dtype=np.int32))
        shape = (len(self.user_ids), len(self.item_ids))
        self.matrix = csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)), shape=shape)
        self.matrix.sum_duplicates()
        self.matrix.data[:] = 1.0
        self.matrix.sort_indices()
        self.item_matrix = self.matrix.T.tocsr()
        self.item_matrix.sort_indices()
        self.popularity = np.asarray(self.matrix.sum(axis=0)).ravel().astype(np.float32)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (shape[0], self.factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.01, (shape[1], self.factors)).astype(np.float32)

        for iteration in range(self.iterations):
            start = time.perf_counter()
            gram = (self.item_factors.T @ self.item_factors).astype(np.float32)
            _cg_update(self.matrix.indptr, self.matrix.indices, self.item_factors,
                       self.user_factors, gram, self.alpha, self.regularization, self.cg_steps)
            gram = (self.user_factors.T @ self.user_factors).astype(np.float32)
            _cg_update(self.item_matrix.indptr, self.item_matrix.indices, self.user_factors,
                       self.item_factors, gram, self.alpha, self.regularization, self.cg_steps)
            print(f'ALS iteration {iteration + 1}/{self.iterations}: {time.perf_counter() - start:.1f}s', flush=True)
        return self

    def recommend(self, requested_users, weights=(0.0,), top_k=10, batch_size=256):
        """为每个权重返回 {user_id: [原始 item_id]}；权重为热门度占比。"""
        requested_users = np.asarray(list(requested_users), dtype=np.int32)
        positions = np.searchsorted(self.user_ids, requested_users)
        if np.any(positions >= len(self.user_ids)) or np.any(self.user_ids[positions] != requested_users):
            raise ValueError('请求中有训练数据未出现的用户')
        weights = tuple(float(w) for w in weights)
        if any(w < 0 or w > 1 for w in weights):
            raise ValueError('热门度融合权重必须在 0 到 1 之间')
        pop = np.log1p(self.popularity)
        pop = (pop - pop.mean()) / max(pop.std(), 1e-6)
        results = {weight: {} for weight in weights}
        for start in range(0, len(requested_users), batch_size):
            end = min(start + batch_size, len(requested_users))
            batch_pos = positions[start:end]
            scores = self.user_factors[batch_pos] @ self.item_factors.T
            if any(w > 0 for w in weights):
                means = scores.mean(axis=1, keepdims=True)
                stds = np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
                normalized = (scores - means) / stds
            else:
                normalized = scores
            for weight in weights:
                combined = normalized.copy() if weight == 0 else (1 - weight) * normalized + weight * pop
                for local, user_pos in enumerate(batch_pos):
                    seen = self.matrix.indices[self.matrix.indptr[user_pos]:self.matrix.indptr[user_pos + 1]]
                    combined[local, seen] = -np.inf
                if np.any(np.isfinite(combined).sum(axis=1) < top_k):
                    raise ValueError('某位用户的未读候选不足 Top-K')
                best = np.argpartition(combined, -top_k, axis=1)[:, -top_k:]
                best_scores = np.take_along_axis(combined, best, axis=1)
                best = np.take_along_axis(best, np.argsort(-best_scores, axis=1), axis=1)
                for local, user_id in enumerate(requested_users[start:end]):
                    results[weight][int(user_id)] = self.item_ids[best[local]].astype(int).tolist()
            if end % 8192 < batch_size or end == len(requested_users):
                print(f'Ranking {end}/{len(requested_users)} users', flush=True)
        return results


def _targets_and_history(model, validation):
    targets = defaultdict(set)
    for uid, iid in zip(validation.user_id.to_numpy(), validation.item_id.to_numpy()):
        targets[int(uid)].add(int(iid))
    history = {}
    for row, uid in enumerate(model.user_ids):
        indices = model.matrix.indices[model.matrix.indptr[row]:model.matrix.indptr[row + 1]]
        history[int(uid)] = set(model.item_ids[indices].astype(int).tolist())
    return dict(targets), history


def _save_recommendations(recommendations, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = ((uid, iid) for uid, items in recommendations.items() for iid in items)
    pd.DataFrame(rows, columns=['user_id', 'item_id']).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Implicit ALS for the book recommendation task')
    parser.add_argument('mode', choices=['validate', 'submit'])
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=40.0)
    parser.add_argument('--regularization', type=float, default=0.1)
    parser.add_argument('--cg-steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pop-weight', type=float, default=0.0,
                        help='提交时使用的热门度融合权重')
    parser.add_argument('--train-path', help='覆盖默认训练文件路径，供额外切分复测')
    parser.add_argument('--val-path', help='覆盖默认验证文件路径，供额外切分复测')
    parser.add_argument('--output-tag', default='', help='验证输出文件后缀，避免覆盖主切分结果')
    args = parser.parse_args()
    if args.factors <= 0 or args.iterations <= 0 or args.cg_steps <= 0:
        parser.error('factors、iterations 和 cg-steps 必须为正整数')

    train_path = args.train_path or ('outputs/train_split.csv' if args.mode == 'validate' else 'data/train_dataset.csv')
    train = pd.read_csv(train_path, dtype={'user_id': 'int32', 'item_id': 'int32'})
    model = ImplicitALS(args.factors, args.iterations, args.alpha,
                        args.regularization, args.cg_steps, args.seed)
    started = time.perf_counter()
    model.fit(train)
    train_seconds = time.perf_counter() - started
    if args.mode == 'validate':
        validation = pd.read_csv(args.val_path or 'outputs/val_split.csv', dtype={'user_id': 'int32', 'item_id': 'int32'})
        targets, history = _targets_and_history(model, validation)
        weights = (0.0, 0.1, 0.25, 0.5)
        recs_by_weight = model.recommend(sorted(targets), weights=weights)
        evaluations = {}
        for weight, recs in recs_by_weight.items():
            result = evaluate(recs, targets, user_train_items=history)
            result['coverage'] = len({iid for items in recs.values() for iid in items}) / len(model.item_ids)
            evaluations[str(weight)] = result
            print(f'pop_weight={weight}: micro-F1={result["micro_f1"]:.6f}, macro-F1={result["macro_f1"]:.6f}, coverage={result["coverage"]:.4f}')
        best_weight = max(weights, key=lambda w: evaluations[str(w)]['micro_f1'])
        suffix = f'_{args.output_tag}' if args.output_tag else ''
        recs_path = f'outputs/c_best_recs{suffix}.csv'
        result_path = f'outputs/c_als_validation{suffix}.json'
        _save_recommendations(recs_by_weight[best_weight], recs_path)
        print_result(evaluations[str(best_weight)])
        payload = {'mode': 'validate', 'parameters': vars(args), 'train_seconds': train_seconds,
                   'best_pop_weight': best_weight, 'evaluations': evaluations}
        os.makedirs('outputs', exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f'Best weight={best_weight}; saved {result_path} and {recs_path}')
    else:
        test = pd.read_csv('data/test_dataset.csv', dtype={'user_id': 'int32'})
        recs = model.recommend(test.user_id, weights=(args.pop_weight,))[args.pop_weight]
        _save_recommendations(recs, 'data/submission_final.csv')
        print(f'Saved data/submission_final.csv ({len(recs) * 10} rows); train_seconds={train_seconds:.1f}')


if __name__ == '__main__':
    main()
