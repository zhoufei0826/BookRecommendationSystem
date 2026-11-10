"""
成员 A 任务 8：实现贝叶斯个性化排序（BPR）模型，作为对 ALS 的排序对照
输出：outputs/bpr_recs.csv, outputs/bpr_eval.txt, outputs/bpr_model_info.txt

实现说明：
- 基于矩阵分解的 BPR (BPR-MF)，使用 numpy 纯手写实现
- 预测分数：x_ui = p_u · q_i
- 训练目标：max ln σ(x_ui - x_uj)，其中 i 是正样本，j 是负样本（未交互物品）
- 用于隐式反馈数据，无需明确负反馈，通过负采样构建排序对
"""

import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict
from evaluate import load_val_targets, evaluate, print_result, save_result

TRAIN_SPLIT_PATH = os.path.join('outputs', 'train_split.csv')
VAL_SPLIT_PATH = os.path.join('outputs', 'val_split.csv')
OUTPUT_DIR = 'outputs'
RECS_PATH = os.path.join(OUTPUT_DIR, 'bpr_recs.csv')
EVAL_PATH = os.path.join(OUTPUT_DIR, 'bpr_eval.txt')
MODEL_INFO_PATH = os.path.join(OUTPUT_DIR, 'bpr_model_info.txt')

# 超参数
N_FACTORS = 32
N_EPOCHS = 20
BATCH_SIZE = 10000
LEARNING_RATE = 0.05
REG_LAMBDA = 0.01
NEGATIVE_SAMPLES = 1
RANDOM_SEED = 42
TOP_K = 10


class BPRMF:
    def __init__(self, n_factors=N_FACTORS, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, reg_lambda=REG_LAMBDA,
                 n_negative=NEGATIVE_SAMPLES, random_seed=RANDOM_SEED):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_negative = n_negative
        self.random_seed = random_seed
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.P = None
        self.Q = None
        self.n_users = 0
        self.n_items = 0
    def _init_mappings(self, train_df):
        users = sorted(train_df['user_id'].unique())
        items = sorted(train_df['item_id'].unique())
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.n_users = len(users)
        self.n_items = len(items)

    def _init_factors(self):
        rng = np.random.RandomState(self.random_seed)
        self.P = rng.normal(0, 0.1, size=(self.n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, size=(self.n_items, self.n_factors))

    def fit(self, train_df):
        self._init_mappings(train_df)
        self._init_factors()

        users = train_df['user_id'].map(self.user_to_idx).to_numpy(dtype=np.int32)
        items = train_df['item_id'].map(self.item_to_idx).to_numpy(dtype=np.int32)
        user_pos_items = [set() for _ in range(self.n_users)]
        for user_idx, item_idx in zip(users, items):
            user_pos_items[user_idx].add(item_idx)

        rng = np.random.RandomState(self.random_seed)
        if len(users) == 0:
            raise ValueError("BPR 训练数据不能为空")
        if self.n_negative <= 0:
            raise ValueError("n_negative 必须为正整数")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")

        for epoch in range(self.n_epochs):
            order = rng.permutation(len(users))
            total_loss = 0.0
            n_updates = 0

            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                user_batch = np.repeat(users[batch], self.n_negative)
                positive_batch = np.repeat(items[batch], self.n_negative)
                negative_batch = rng.randint(self.n_items, size=len(user_batch))

                invalid = np.fromiter(
                    (negative in user_pos_items[user] for user, negative in zip(user_batch, negative_batch)),
                    dtype=bool,
                    count=len(user_batch),
                )
                attempts = 0
                while invalid.any() and attempts < 100:
                    negative_batch[invalid] = rng.randint(self.n_items, size=int(invalid.sum()))
                    invalid = np.fromiter(
                        (negative in user_pos_items[user] for user, negative in zip(user_batch, negative_batch)),
                        dtype=bool,
                        count=len(user_batch),
                    )
                    attempts += 1
                if invalid.any():
                    raise RuntimeError("无法为部分用户采样未交互图书")

                user_factors = self.P[user_batch].copy()
                positive_factors = self.Q[positive_batch].copy()
                negative_factors = self.Q[negative_batch].copy()
                differences = np.sum(
                    user_factors * (positive_factors - negative_factors),
                    axis=1,
                )
                sigmoid = 1.0 / (1.0 + np.exp(-np.clip(differences, -60, 60)))
                gradient = 1.0 - sigmoid

                user_gradients = (
                    gradient[:, None] * (positive_factors - negative_factors)
                    - self.reg_lambda * user_factors
                )
                positive_gradients = (
                    gradient[:, None] * user_factors
                    - self.reg_lambda * positive_factors
                )
                negative_gradients = (
                    -gradient[:, None] * user_factors
                    - self.reg_lambda * negative_factors
                )
                np.add.at(self.P, user_batch, self.learning_rate * user_gradients)
                np.add.at(self.Q, positive_batch, self.learning_rate * positive_gradients)
                np.add.at(self.Q, negative_batch, self.learning_rate * negative_gradients)

                total_loss += float(-np.log(np.maximum(sigmoid, 1e-10)).sum())
                n_updates += len(user_batch)

            avg_loss = total_loss / max(n_updates, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, 平均损失: {avg_loss:.4f}")

        print("BPR 训练完成。")

    def recommend_all(self, user_ids, user_train_items, top_k=TOP_K):
        recommendations = {}
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        for u in user_ids:
            if u not in self.user_to_idx:
                recommendations[u] = []
                continue
            u_idx = self.user_to_idx[u]
            scores = np.dot(self.Q, self.P[u_idx])
            seen = user_train_items.get(u, set())
            seen_indices = [self.item_to_idx[item_id] for item_id in seen if item_id in self.item_to_idx]
            scores[seen_indices] = -np.inf
            candidate_count = int(np.isfinite(scores).sum())
            candidate_k = min(top_k, candidate_count)
            if candidate_k == 0:
                recommendations[u] = []
                continue
            best_indices = np.argpartition(scores, -candidate_k)[-candidate_k:]
            best_indices = best_indices[np.argsort(scores[best_indices])[::-1]]
            recommendations[u] = [self.idx_to_item[int(idx)] for idx in best_indices]
        return recommendations


def get_user_train_items(train_split_df):
    user_train_items = defaultdict(set)
    for _, row in train_split_df.iterrows():
        user_train_items[int(row['user_id'])].add(int(row['item_id']))
    return dict(user_train_items)


def save_recommendations(recommendations, path=RECS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for user_id, items in recommendations.items():
        for item_id in items:
            rows.append({'user_id': user_id, 'item_id': item_id})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"BPR 推荐结果已保存: {path} ({len(df)} 行)")


def save_model_info(model, path=MODEL_INFO_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    info = {
        'model': 'BPR-MF (Bayesian Personalized Ranking Matrix Factorization)',
        'n_factors': model.n_factors,
        'n_epochs': model.n_epochs,
        'learning_rate': model.learning_rate,
        'reg_lambda': model.reg_lambda,
        'negative_samples': model.n_negative,
        'random_seed': model.random_seed,
        'n_users': model.n_users,
        'n_items': model.n_items,
        'description': '纯 numpy 实现的 BPR 矩阵分解，用于隐式反馈排序，作为 ALS 的对照模型'
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"BPR 模型信息已保存: {path}")


def main():
    train_split = pd.read_csv(TRAIN_SPLIT_PATH)
    val_targets = load_val_targets(VAL_SPLIT_PATH)

    print(f"BPR 训练数据: {len(train_split)} 行，{train_split['user_id'].nunique()} 位用户，{train_split['item_id'].nunique()} 本图书")

    model = BPRMF()
    model.fit(train_split)

    user_train_items = get_user_train_items(train_split)

    all_users = sorted(list(val_targets.keys()))
    recommendations = model.recommend_all(all_users, user_train_items, top_k=TOP_K)

    save_recommendations(recommendations)
    save_model_info(model)

    result = evaluate(
        recommendations,
        val_targets,
        top_k=TOP_K,
        user_train_items=user_train_items,
    )
    print_result(result)
    save_result(result, path=EVAL_PATH)

    print("成员 A 任务 8 完成：BPR 模型已实现并评估。")


if __name__ == '__main__':
    main()