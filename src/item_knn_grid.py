"""
成员 B 补充实验：ItemKNN 平台化参数搜索与多样性评估

目的（补齐前期调参过浅、相似度单一的问题）：
1. 相似度变体对比：余弦 / 余弦+相似度收缩 / 条件概率 / TF-IDF 余弦 / 用户侧加权余弦
2. 在最优相似度上扫描 beta（历史降权）、alpha（候选惩罚）、N（邻居截断）
3. 消融 hist_norm（历史权重归一化），把归一化与 beta 解耦
4. 多样性约束实验：排除全局最热门图书，观察 F1 与覆盖率/新颖度的权衡
5. 每次实验同时记录覆盖率、平均热门度分位、新颖度、列表内多样性（ILD）

输出：outputs/itemknn_grid_results.csv（全部结果，按 micro-F1 排序）、控制台 Top 榜

用法（在项目根目录执行）：
    python src/item_knn_grid.py
    python src/item_knn_grid.py --stage1-limit 3
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from evaluate import load_val_targets, evaluate
from item_knn import (
    build_matrix,
    compute_similarity,
    get_user_train_items,
    load_train_split,
    recommend_all,
    truncate_neighbors,
)

TRAIN_SPLIT_PATH = os.path.join('outputs', 'train_split.csv')
VAL_SPLIT_PATH = os.path.join('outputs', 'val_split.csv')
RESULT_PATH = os.path.join('outputs', 'itemknn_grid_results.csv')
TOP_K = 10


# ------------------------------------------------------------ 多样性指标
def compute_diversity(recs, pop, total_pop, sim=None, n_sample_users=3000, seed=42):
    """覆盖率、平均热门度分位、新颖度、列表内平均相似度（ILD）"""
    rec_items = recs['item_id'].to_numpy()
    coverage = len(np.unique(rec_items)) / len(pop)

    # 热门度分位：0 = 最冷门，1 = 最热门
    order = np.argsort(np.argsort(pop))  # 升序排名
    rank_pct = order / max(len(pop) - 1, 1)
    rank_pct = np.where(pop > 0, rank_pct, 0.0)
    avg_pop_pct = float(rank_pct[rec_items].mean())

    # 新颖度：-log2(p_i)，p_i 为图书交互占比
    p = pop / max(total_pop, 1)
    novelty = float(np.mean(-np.log2(np.clip(p[rec_items], 1e-12, None))))

    ild = float('nan')
    if sim is not None:
        rng = np.random.default_rng(seed)
        users = np.unique(recs['user_id'].to_numpy())
        sample = rng.choice(users, size=min(n_sample_users, len(users)), replace=False)
        grouped = recs[recs['user_id'].isin(sample)].groupby('user_id')['item_id'].apply(
            lambda s: s.to_numpy())
        vals = []
        for items in grouped:
            if len(items) < 2:
                continue
            sub = sim[items][:, items].toarray()
            n = len(items)
            vals.append((sub.sum() - np.trace(sub)) / (n * (n - 1)))
        ild = float(np.mean(vals)) if vals else float('nan')

    return {
        'coverage': coverage,
        'avg_pop_pct': avg_pop_pct,
        'novelty': novelty,
        'ild': ild,
        'distinct_items': int(len(np.unique(rec_items))),
    }


# ------------------------------------------------------------ 实验执行
class Runner:
    def __init__(self, train_df, user_to_idx, item_to_idx, mat,
                 val_targets, user_train_items, popularity):
        self.train_df = train_df
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.mat = mat
        self.val_targets = val_targets
        self.user_train_items = user_train_items
        self.popularity = popularity

        # 图书热度数组（按 item_to_idx 顺序）
        self.pop = np.zeros(len(item_to_idx), dtype=np.float64)
        for item_id, cnt in popularity.items():
            self.pop[item_to_idx[item_id]] = cnt
        self.total_pop = float(self.pop.sum())

        self._sim_cache = {}
        self._nb_cache = {}
        self.rows = []

    # ---- 缓存
    def sim(self, mode, shrinkage=0.0, user_weight=0.0):
        key = (mode, shrinkage, user_weight)
        if key not in self._sim_cache:
            t0 = time.time()
            self._sim_cache[key] = compute_similarity(
                self.mat, mode, shrinkage=shrinkage, user_weight=user_weight)
            print(f"  [sim] {key} 计算完成 {time.time() - t0:.1f}s")
        return self._sim_cache[key]

    def neighbors(self, mode, n, shrinkage=0.0, user_weight=0.0):
        key = (mode, n, shrinkage, user_weight)
        if key not in self._nb_cache:
            t0 = time.time()
            self._nb_cache[key] = truncate_neighbors(
                self.sim(mode, shrinkage, user_weight), n)
            print(f"  [nb] {key} 截断完成 {time.time() - t0:.1f}s")
        return self._nb_cache[key]

    # ---- 单次实验
    def run(self, label, mode, n, beta, alpha=0.0, shrinkage=0.0,
            user_weight=0.0, hist_norm='sum', max_pop_share=None, note=''):
        nb = self.neighbors(mode, n, shrinkage, user_weight)
        t0 = time.time()
        recommendations, fallback = recommend_all(
            nb, self.user_train_items, self.popularity, self.user_to_idx,
            self.item_to_idx, beta=beta, alpha=alpha, top_k=TOP_K,
            hist_norm=hist_norm, max_pop_share=max_pop_share)
        runtime = time.time() - t0

        result = evaluate(recommendations, self.val_targets, top_k=TOP_K,
                          user_train_items=self.user_train_items)

        recs_df = pd.DataFrame(
            [(u, i) for u, items in recommendations.items() for i in items],
            columns=['user_id', 'item_id'])
        # ILD 用余弦相似度衡量，避免不同相似度变体口径不一致
        div = compute_diversity(recs_df, self.pop, self.total_pop,
                                sim=self.sim('cosine'))

        row = {
            '实验': label,
            '相似度': mode,
            'shrinkage': shrinkage,
            'user_weight': user_weight,
            'N': n,
            'beta': beta,
            'alpha': alpha,
            'hist_norm': hist_norm,
            'max_pop_share': max_pop_share,
            'micro_P': result['micro_precision'],
            'micro_R': result['micro_recall'],
            'micro_F1': result['micro_f1'],
            'macro_F1': result['macro_f1'],
            '覆盖率': div['coverage'],
            '平均热门度分位': div['avg_pop_pct'],
            '新颖度': div['novelty'],
            'ILD': div['ild'],
            '回退用户数': fallback,
            '运行时间': round(runtime, 1),
            '备注': note,
        }
        self.rows.append(row)
        print(f"    {label}: micro-F1={result['micro_f1']:.6f} "
              f"P={result['micro_precision']:.6f} 覆盖率={div['coverage']:.3f} "
              f"热门度分位={div['avg_pop_pct']:.3f} ({runtime:.1f}s)")
        return row

    def best(self, rows):
        return max(rows, key=lambda r: r['micro_F1'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-limit', type=int, default=None,
                        help='仅跑前 k 个相似度变体（调试用）')
    args = parser.parse_args()

    print("=" * 70)
    print("ItemKNN 参数搜索与多样性评估")
    print("=" * 70)

    train_df, user_to_idx, item_to_idx = load_train_split(TRAIN_SPLIT_PATH)
    mat = build_matrix(train_df, user_to_idx, item_to_idx)
    val_targets = load_val_targets(VAL_SPLIT_PATH)
    user_train_items = get_user_train_items(train_df)
    popularity = train_df['item_id'].value_counts().to_dict()
    print(f"训练: {len(train_df)} 行 / {len(user_to_idx)} 用户 / {len(item_to_idx)} 图书")

    runner = Runner(train_df, user_to_idx, item_to_idx, mat,
                    val_targets, user_train_items, popularity)

    # ---------------- 阶段 1：相似度变体（N=100, beta=0.5, hist_norm=sum）
    print("\n[阶段 1] 相似度变体对比（N=100, beta=0.5）")
    variants = [
        ('S1-余弦', 'cosine', 0.0, 0.0, '基准：按度归一，抑制热门书偏置'),
        ('S2-余弦+收缩λ=10', 'cosine', 10.0, 0.0, '共同用户少时相似度向 0 收缩'),
        ('S3-余弦+收缩λ=25', 'cosine', 25.0, 0.0, '更强收缩，压长尾噪声'),
        ('S4-条件概率', 'conditional', 0.0, 0.0, 'P(i|j) 对称化'),
        ('S5-TFIDF余弦', 'tfidf', 0.0, 0.0, 'IDF 抑制热门书权重'),
        ('S6-余弦+用户加权γ=0.5', 'cosine', 0.0, 0.5, '1/|u|^0.5 降低活跃用户影响'),
    ]
    if args.stage1_limit:
        variants = variants[:args.stage1_limit]

    stage1 = []
    for label, mode, shrink, uw, note in variants:
        stage1.append(runner.run(label, mode, 100, 0.5,
                                 shrinkage=shrink, user_weight=uw, note=note))
    best1 = runner.best(stage1)
    best_mode = best1['相似度']
    best_shrink = best1['shrinkage']
    best_uw = best1['user_weight']
    print(f"  → 阶段 1 最优: {best1['实验']} (micro-F1={best1['micro_F1']:.6f})")

    # ---------------- 阶段 2：beta 扫描
    print("\n[阶段 2] beta 扫描（历史热门书降权）")
    stage2 = [best1]
    for beta in (0.0, 0.3, 0.7):
        stage2.append(runner.run(f'S7-beta={beta}', best_mode, 100, beta,
                                 shrinkage=best_shrink, user_weight=best_uw,
                                 note='beta 扫描'))
    best2 = runner.best(stage2)
    best_beta = best2['beta']
    print(f"  → 阶段 2 最优: {best2['实验']} beta={best_beta} "
          f"(micro-F1={best2['micro_F1']:.6f})")

    # ---------------- 阶段 3：hist_norm 消融（与 beta 解耦）
    print("\n[阶段 3] hist_norm 消融（beta 固定为最优值）")
    stage3 = [best2]
    stage3.append(runner.run('S8-不归一化', best_mode, 100, best_beta,
                             shrinkage=best_shrink, user_weight=best_uw,
                             hist_norm='none',
                             note='历史权重不归一化，隔离归一化影响'))
    best3 = runner.best(stage3)
    print(f"  → 阶段 3 最优: {best3['实验']} hist_norm={best3['hist_norm']}")

    # ---------------- 阶段 4：alpha 与 N
    print("\n[阶段 4] alpha 与邻居数 N")
    stage4 = [best3]
    for alpha in (0.2, 0.4):
        stage4.append(runner.run(f'S9-alpha={alpha}', best_mode, 100, best_beta,
                                 alpha=alpha, shrinkage=best_shrink, user_weight=best_uw,
                                 hist_norm=best3['hist_norm'], note='候选热度惩罚'))
    for n in (50, 200):
        stage4.append(runner.run(f'S10-N={n}', best_mode, n, best_beta,
                                 shrinkage=best_shrink, user_weight=best_uw,
                                 hist_norm=best3['hist_norm'], note='邻居数扫描'))
    best4 = runner.best(stage4)
    print(f"  → 阶段 4 最优: {best4['实验']} (micro-F1={best4['micro_F1']:.6f})")

    # ---------------- 阶段 5：多样性约束
    print("\n[阶段 5] 多样性约束（排除最热门 1% 图书）")
    stage5 = [best4]
    stage5.append(runner.run('S11-屏蔽最热1%', best_mode, best4['N'], best4['beta'],
                             alpha=best4['alpha'], shrinkage=best_shrink,
                             user_weight=best_uw, hist_norm=best4['hist_norm'],
                             max_pop_share=0.01,
                             note='多样性实验：牺牲部分 F1 换取覆盖与新颖度'))

    # ---------------- 汇总
    df = pd.DataFrame(runner.rows)
    df = df.sort_values('micro_F1', ascending=False).reset_index(drop=True)
    df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存: {RESULT_PATH}")

    print("\n" + "=" * 70)
    print("Top 8（按 micro-F1@10）")
    print("=" * 70)
    cols = ['实验', '相似度', 'N', 'beta', 'alpha', 'micro_F1', 'macro_F1',
            '覆盖率', '平均热门度分位', 'ILD']
    print(df[cols].head(8).to_string(index=False,
                                     float_format=lambda x: f"{x:.4f}"))

    best = df.iloc[0]
    print("\n最佳配置：")
    for k in ['实验', '相似度', 'shrinkage', 'user_weight', 'N', 'beta', 'alpha',
              'hist_norm', 'micro_F1', 'macro_F1', '覆盖率', 'novelty'.replace('novelty', '新颖度')]:
        if k in best:
            print(f"  {k}: {best[k]}")
    print("\n复现最佳配置的命令：")
    print(f"  python src/item_knn.py --sim {best['相似度']} --neighbors {int(best['N'])} "
          f"--beta {best['beta']} --alpha {best['alpha']} "
          f"--shrinkage {best['shrinkage']} --user-weight {best['user_weight']} "
          f"--hist-norm {best['hist_norm']}")


if __name__ == '__main__':
    main()
