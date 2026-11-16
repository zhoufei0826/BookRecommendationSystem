"""
成员 B 任务 2-4：基于物品的协同过滤（ItemKNN）推荐实现与验证

实现说明：
- 相似度：余弦相似度（对热门图书按度归一，抑制热门书偏置），另可切 raw 共现做对比
- 邻居截断：每本书仅保留相似度最高的 Top-N 邻居，避免构造全量稠密物品对矩阵
- 候选打分：score(j) = Σ_{i∈history(u)} pop_i^(-beta) * sim(i,j)，可选再乘 pop_j^(-alpha)
  beta > 0 时降低热门历史书的话语权；alpha > 0 时对热门候选做惩罚
- 已读过滤：过滤用户训练部分已交互图书；不足 Top-K 时用未读全局热门书回退

输入：outputs/train_split.csv, outputs/val_split.csv（成员 A 的固定切分，Seed 42）
输出：outputs/itemknn_recs.csv, outputs/itemknn_eval.txt, outputs/itemknn_model_info.txt
      并在 reports/experiments_B.csv 追加实验记录（B 的独立记录表，不改动成员 A 的文件）

用法（在项目根目录执行）：
    python src/item_knn.py                          # 默认最优配置 N=100, beta=0.5
    python src/item_knn.py --neighbors 100,300 --beta 0.5 --tag B-002
    python src/item_knn.py --neighbors 100 --sim raw --tag B-003
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from evaluate import load_val_targets, evaluate, print_result, save_result

TRAIN_SPLIT_PATH = os.path.join('outputs', 'train_split.csv')
VAL_SPLIT_PATH = os.path.join('outputs', 'val_split.csv')
OUTPUT_DIR = 'outputs'
RECS_PATH = os.path.join(OUTPUT_DIR, 'itemknn_recs.csv')
EVAL_PATH = os.path.join(OUTPUT_DIR, 'itemknn_eval.txt')
MODEL_INFO_PATH = os.path.join(OUTPUT_DIR, 'itemknn_model_info.txt')
EXPERIMENT_PATH = os.path.join('reports', 'experiments_B.csv')

TOP_K = 10
DEFAULT_NEIGHBORS = 100
DEFAULT_BETA = 0.5
DEFAULT_ALPHA = 0.0


# ---------------------------------------------------------------- 数据准备
def load_train_split(path=TRAIN_SPLIT_PATH):
    """读取固定训练切分，返回 (DataFrame, 用户索引映射, 图书索引映射)"""
    df = pd.read_csv(path)
    users = np.sort(df['user_id'].unique())
    items = np.sort(df['item_id'].unique())
    user_to_idx = {int(u): i for i, u in enumerate(users)}
    item_to_idx = {int(i): j for j, i in enumerate(items)}
    return df, user_to_idx, item_to_idx


def get_user_train_items(train_df):
    """每位用户在训练部分已交互的图书集合（与成员 A 的接口保持一致）"""
    return {
        int(user_id): set(items.astype(int))
        for user_id, items in train_df.groupby('user_id')['item_id']
    }


def build_matrix(train_df, user_to_idx, item_to_idx):
    """构建稀疏 用户×图书 01 交互矩阵（CSR）"""
    rows = train_df['user_id'].map(user_to_idx).to_numpy()
    cols = train_df['item_id'].map(item_to_idx).to_numpy()
    mat = csr_matrix(
        (np.ones(len(train_df), dtype=np.float32), (rows, cols)),
        shape=(len(user_to_idx), len(item_to_idx)),
    )
    return mat


# ---------------------------------------------------------------- 相似度
def build_cooccurrence(mat):
    """原始共现计数矩阵 C（items × items），供余弦归一与相似度收缩共用"""
    item_user = mat.T.tocsr()
    return (item_user @ item_user.T).tocsr(), item_user


def compute_similarity(mat, mode='cosine', shrinkage=0.0, user_weight=0.0):
    """图书-图书相似度矩阵（items × items，含自相似，截断阶段会去掉）

    mode:
        raw        - 原始共现次数（对照，受热门书偏置）
        cosine     - 余弦相似度（按图书度归一，抑制热门书偏置）
        conditional- 条件概率 P(i|j) = co(i,j)/pop(j)，再做对称化
        tfidf      - 图书 TF-IDF 加权后求余弦（IDF 抑制热门书）
    shrinkage:
        λ > 0 时按 co/(co+λ) 收缩，共同用户少的图书对相似度向 0 拉，
        缓解长尾图书「仅凭 1~2 个共同用户就得到高相似度」的噪声。
    user_weight:
        γ > 0 时对用户加权 1/|u|^γ，降低超活跃用户对相似度的主导
        （活跃用户几乎和所有书共现，会污染邻居结构）。
    """
    co, item_user = build_cooccurrence(mat)
    pop = np.asarray(item_user.sum(axis=1)).ravel().astype(np.float64)

    if mode == 'raw':
        sim = co
    elif mode == 'conditional':
        inv = 1.0 / np.maximum(pop, 1e-9)
        sim = co.multiply(inv[None, :]).tocsr()      # P(i|j)
        sim = ((sim + sim.T) * 0.5).tocsr()          # 对称化
    elif mode in ('cosine', 'tfidf'):
        iu = item_user.astype(np.float64)
        if user_weight > 0:
            # mat 为 用户×图书，按行求和得到每位用户的交互数
            n_u = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
            uw = 1.0 / np.power(np.maximum(n_u, 1.0), user_weight)
            iu = iu.multiply(uw[None, :]).tocsr()
        if mode == 'tfidf':
            n_users = mat.shape[0]
            idf = np.log1p(n_users / np.maximum(pop, 1.0))
            iu = iu.multiply(idf[:, None]).tocsr()
        deg = np.asarray(iu.sum(axis=1)).ravel()
        norm = np.sqrt(np.maximum(deg, 1e-9))
        iu = iu.multiply(1.0 / norm[:, None]).tocsr()
        sim = (iu @ iu.T).tocsr()
    else:
        raise ValueError(f"未支持的相似度模式: {mode}")

    if shrinkage > 0:
        sh = co.copy()
        sh.data = (sh.data / (sh.data + shrinkage)).astype(np.float32)
        sim = sim.multiply(sh).tocsr()

    return sim


def truncate_neighbors(sim, n_neighbors):
    """每行仅保留 Top-N 相似邻居（去自相似），逐行截断以控制内存"""
    sim = sim.tocsr()
    data_parts, index_parts, indptr = [], [], [0]
    for r in range(sim.shape[0]):
        s, e = sim.indptr[r], sim.indptr[r + 1]
        cols = sim.indices[s:e]
        vals = sim.data[s:e]
        keep = cols != r
        cols, vals = cols[keep], vals[keep]
        if len(cols) > n_neighbors:
            idx = np.argpartition(vals, -n_neighbors)[-n_neighbors:]
            cols, vals = cols[idx], vals[idx]
        order = np.argsort(vals)[::-1]
        index_parts.append(cols[order])
        data_parts.append(vals[order])
        indptr.append(indptr[-1] + len(cols))
    return csr_matrix(
        (np.concatenate(data_parts), np.concatenate(index_parts), indptr),
        shape=sim.shape,
    )


# ---------------------------------------------------------------- 推荐生成
def recommend_all(neighbors, user_train_items, popularity, user_to_idx,
                  item_to_idx, beta=DEFAULT_BETA, alpha=DEFAULT_ALPHA, top_k=TOP_K,
                  hist_norm='sum', max_pop_share=None):
    """对全部训练用户生成 Top-K 推荐，返回 ({user_id: [item_id, ...]}, 回退用户数)

    打分：score(j) = Σ_{i∈history(u)} w_i · sim(i,j)
        w_i = pop_i^(-beta)              beta>0 时降低热门历史书的话语权
        hist_norm='sum' 时对 w 做归一化，使不同活跃度用户得分尺度可比；
        'none' 时不做归一化（活跃用户得分整体更大，但只在用户内部排序用，影响有限）
        alpha>0 时候选分数再乘 pop_j^(-alpha)（热门候选惩罚）
        max_pop_share: 若给定（如 0.01），推荐时排除全局热度前 N% 的图书（多样性约束实验）
    """
    n_items = len(item_to_idx)
    items_arr = np.array(sorted(item_to_idx.keys()))

    pop = np.zeros(n_items, dtype=np.float64)
    for item_id, cnt in popularity.items():
        pop[item_to_idx[item_id]] = cnt

    # 全局热门序（按未读回退用）
    pop_rank = np.argsort(pop)[::-1]
    pop_rank_ids = items_arr[pop_rank]

    # 可选：屏蔽超热门图书（多样性实验）
    blocked = np.zeros(n_items, dtype=bool)
    if max_pop_share is not None:
        n_block = int(n_items * max_pop_share)
        blocked[pop_rank[:n_block]] = True

    # 历史权重与候选惩罚因子
    hist_cache = {}
    for user_id, seen in user_train_items.items():
        idx = np.array([item_to_idx[i] for i in seen if i in item_to_idx], dtype=np.int64)
        if beta > 0 and len(idx) > 0:
            w = pop[idx] ** (-beta)
            if hist_norm == 'sum':
                wsum = w.sum()
                w = w / wsum if wsum > 0 else w
        else:
            w = np.ones(len(idx), dtype=np.float64)
        hist_cache[user_id] = (idx, w)

    cand_pen = pop ** (-alpha) if alpha > 0 else None

    recommendations = {}
    fallback_users = 0
    for user_id in user_to_idx:
        seen = user_train_items.get(user_id, set())
        idx, w = hist_cache[user_id]
        if len(idx) == 0:
            recs = [int(i) for i in pop_rank_ids
                    if i not in seen and not blocked[item_to_idx[int(i)]]][:top_k]
            fallback_users += 1
            recommendations[user_id] = recs
            continue

        rows = neighbors[idx]
        scores = np.asarray(rows.multiply(w[:, None]).sum(axis=0)).ravel()
        if cand_pen is not None:
            scores = scores * cand_pen
        scores[idx] = -1.0  # 已读过滤
        if max_pop_share is not None:
            scores[blocked] = -1.0

        n_positive = int(np.count_nonzero(scores > 0))
        if n_positive >= top_k:
            cand = np.argpartition(scores, -top_k)[-top_k:]
            cand = cand[np.argsort(scores[cand])[::-1]]
        else:
            cand = np.argsort(scores)[::-1][:top_k]

        recs = [int(items_arr[c]) for c in cand if scores[c] > 0]
        # 回退：未读全局热门书补足
        if len(recs) < top_k:
            fallback_users += 1
            chosen = set(recs)
            for item_id in pop_rank_ids:
                if len(recs) >= top_k:
                    break
                if item_id not in seen and int(item_id) not in chosen:
                    recs.append(int(item_id))
                    chosen.add(int(item_id))
        recommendations[user_id] = recs[:top_k]

    return recommendations, fallback_users


# ---------------------------------------------------------------- 输出
def save_recommendations(recommendations, path=RECS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [{'user_id': uid, 'item_id': iid}
            for uid, items in recommendations.items() for iid in items]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"推荐结果已保存: {path} ({len(df)} 行)")
    return df


def append_experiment(split_name, model, params, result, coverage, runtime_s,
                      out_recs, note):
    """按成员 A 的 experiments.csv 表头追加一条实验记录，编号自动递增"""
    os.makedirs(os.path.dirname(EXPERIMENT_PATH), exist_ok=True)
    fields = ['实验编号', '负责人', '数据切分', '模型', '关键参数',
              'micro-P@10', 'micro-R@10', 'micro-F1@10', 'macro-F1@10',
              '覆盖率', '运行时间(秒)', '备注']

    existing_ids = []
    if os.path.exists(EXPERIMENT_PATH):
        with open(EXPERIMENT_PATH, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                eid = (row.get('实验编号') or '').strip()
                if eid.startswith('B-'):
                    try:
                        existing_ids.append(int(eid.split('-')[1]))
                    except ValueError:
                        pass
    next_id = max(existing_ids) + 1 if existing_ids else 1
    exp_id = params.get('实验编号') or f"B-{next_id:03d}"

    write_header = not os.path.exists(EXPERIMENT_PATH)
    # 已有文件末尾若无换行，先补一个，避免新记录粘到上一行
    if not write_header:
        with open(EXPERIMENT_PATH, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                with open(EXPERIMENT_PATH, 'a', encoding='utf-8-sig') as fa:
                    fa.write('\n')

    with open(EXPERIMENT_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({
            '实验编号': exp_id,
            '负责人': '成员 B',
            '数据切分': split_name,
            '模型': model,
            '关键参数': params.get('关键参数', ''),
            'micro-P@10': f"{result['micro_precision']:.6f}",
            'micro-R@10': f"{result['micro_recall']:.6f}",
            'micro-F1@10': f"{result['micro_f1']:.6f}",
            'macro-F1@10': f"{result['macro_f1']:.6f}",
            '覆盖率': f"{coverage:.4f}",
            '运行时间(秒)': f"{runtime_s:.1f}",
            '备注': note,
        })
    print(f"实验记录已追加: {EXPERIMENT_PATH}（{exp_id}）")
    return exp_id


def save_model_info(info, path=MODEL_INFO_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"模型信息已保存: {path}")


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description='成员 B：ItemKNN 协同过滤')
    parser.add_argument('--neighbors', default=str(DEFAULT_NEIGHBORS),
                        help='邻居截断数，可逗号分隔多个值，如 100,300')
    parser.add_argument('--sim', default='cosine',
                        choices=['cosine', 'raw', 'conditional', 'tfidf'],
                        help='相似度：cosine（余弦，默认）/ raw（原始共现）/ '
                             'conditional（条件概率）/ tfidf（TF-IDF 余弦）')
    parser.add_argument('--shrinkage', type=float, default=0.0,
                        help='相似度收缩强度 λ，按 co/(co+λ) 收缩，0 表示不收缩')
    parser.add_argument('--user-weight', type=float, default=0.0,
                        help='用户侧加权指数 γ，按 1/|u|^γ 降低活跃用户影响，0 表示不加权')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA,
                        help='历史图书热度降权指数，0 表示不加权')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='候选图书热度惩罚指数，0 表示不惩罚')
    parser.add_argument('--hist-norm', default='sum', choices=['sum', 'none'],
                        help='历史权重是否归一化（与 beta 解耦，便于消融）')
    parser.add_argument('--max-pop-share', type=float, default=None,
                        help='多样性实验：排除全局最热门的该比例图书（如 0.01）')
    parser.add_argument('--tag', default='', help='实验标签，用于输出文件命名')
    parser.add_argument('--no-log', action='store_true', help='不写入 experiments_B.csv')
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("成员 B：ItemKNN 协同过滤推荐")
    print("=" * 60)

    train_df, user_to_idx, item_to_idx = load_train_split()
    print(f"训练切分: {len(train_df)} 行，{len(user_to_idx)} 位用户，{len(item_to_idx)} 本图书")

    val_targets = load_val_targets(VAL_SPLIT_PATH)
    print(f"验证目标: {len(val_targets)} 位用户，{sum(len(v) for v in val_targets.values())} 条")

    user_train_items = get_user_train_items(train_df)
    popularity = train_df['item_id'].value_counts().to_dict()

    mat = build_matrix(train_df, user_to_idx, item_to_idx)
    t_sim0 = time.time()
    sim = compute_similarity(mat, args.sim, shrinkage=args.shrinkage,
                             user_weight=args.user_weight)
    print(f"相似度矩阵（{args.sim}, shrinkage={args.shrinkage}, "
          f"user_weight={args.user_weight}）: nnz={sim.nnz}，耗时 {time.time() - t_sim0:.1f}s")

    n_list = [int(x) for x in str(args.neighbors).split(',')]
    all_info = []

    for n in n_list:
        t0 = time.time()
        # 指定 tag 时输出带后缀，避免不同实验互相覆盖；无 tag 时写标准文件名
        suffix = f"_{args.tag}_n{n}" if args.tag else ""
        recs_path = RECS_PATH.replace('.csv', suffix + '.csv')
        eval_path = EVAL_PATH.replace('.txt', suffix + '.txt')

        neighbors = truncate_neighbors(sim, n)
        recommendations, fallback_users = recommend_all(
            neighbors, user_train_items, popularity, user_to_idx, item_to_idx,
            beta=args.beta, alpha=args.alpha, top_k=TOP_K,
            hist_norm=args.hist_norm, max_pop_share=args.max_pop_share,
        )
        rec_df = save_recommendations(recommendations, recs_path)
        result = evaluate(recommendations, val_targets, top_k=TOP_K,
                          user_train_items=user_train_items)
        print_result(result)
        save_result(result, path=eval_path)

        coverage = rec_df['item_id'].nunique() / len(item_to_idx)
        runtime = time.time() - t0

        key_params = (f"sim={args.sim};shrinkage={args.shrinkage};"
                      f"user_weight={args.user_weight};neighbors={n};"
                      f"beta={args.beta};alpha={args.alpha};hist_norm={args.hist_norm}")
        if args.max_pop_share is not None:
            key_params += f";max_pop_share={args.max_pop_share}"
        params = {
            '关键参数': key_params,
            '实验编号': args.tag if args.tag.startswith('B-') else None,
        }
        exp_id = None
        if not args.no_log:
            exp_id = append_experiment(
                'train_split/seed42', 'ItemKNN', params, result, coverage, runtime,
                recs_path,
                f"相似度={args.sim}，收缩λ={args.shrinkage}，用户加权γ={args.user_weight}，"
                f"邻居={n}，历史降权beta={args.beta}，候选惩罚alpha={args.alpha}，"
                f"历史归一化={args.hist_norm}，回退用户数={fallback_users}")

        all_info.append({
            'experiment_id': exp_id,
            'similarity': args.sim,
            'shrinkage': args.shrinkage,
            'user_weight': args.user_weight,
            'n_neighbors': n,
            'beta': args.beta,
            'alpha': args.alpha,
            'hist_norm': args.hist_norm,
            'max_pop_share': args.max_pop_share,
            'micro_f1': result['micro_f1'],
            'macro_f1': result['macro_f1'],
            'coverage': coverage,
            'fallback_users': fallback_users,
            'runtime_s': round(runtime, 1),
        })

    save_model_info({
        'model': 'ItemKNN',
        'split': 'outputs/train_split.csv (Seed 42)',
        'top_k': TOP_K,
        'similarity': args.sim,
        'n_neighbors': n_list,
        'beta': args.beta,
        'alpha': args.alpha,
        'runs': all_info,
        'total_runtime_s': round(time.time() - t_start, 1),
    })

    print("=" * 60)
    print("成员 B 任务 2-4 完成：ItemKNN 已生成推荐并完成统一评估。")
    print("=" * 60)


if __name__ == '__main__':
    main()
