"""
成员 B 补充交付：推荐多样性分析

方案阶段 4 要求「推荐列表中热门图书占比及推荐多样性」，本脚本用统一指标量化：
- 覆盖率：被推荐过的不同图书数 / 全部图书数
- 平均热门度分位：推荐图书在全局热度排名中的平均位置（1 = 最热门）
- 新颖度：推荐图书 -log2(交互占比) 的均值，越高表示越不依赖热门
- 列表内多样性 ILD：同一用户 10 本推荐之间的平均余弦相似度，越低越多样
- 头部图书占比：推荐列表中属于头部（累计贡献前 20% 交互）的比例

输出：outputs/diversity_summary.csv、控制台对比表

用法（在项目根目录执行）：
    python src/diversity_analysis.py
    python src/diversity_analysis.py --recs outputs/itemknn_recs.csv outputs/popularity_baseline_recs.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from item_knn import build_matrix, compute_similarity, load_train_split
from item_knn_grid import compute_diversity

TRAIN_SPLIT_PATH = os.path.join('outputs', 'train_split.csv')
DEFAULT_RECS = [
    os.path.join('outputs', 'itemknn_recs.csv'),
    os.path.join('outputs', 'popularity_baseline_recs.csv'),
]
OUT_PATH = os.path.join('outputs', 'diversity_summary.csv')


def head_share(recs, pop, n_head_ratio=0.20):
    """推荐列表中头部图书（贡献前 20% 交互）的占比"""
    order = np.argsort(pop)[::-1]
    cum = np.cumsum(pop[order]) / max(pop.sum(), 1)
    n_head = int((cum <= n_head_ratio).sum()) + 1
    head = set(order[:n_head])
    return float(np.mean([1.0 if i in head else 0.0 for i in recs['item_id'].to_numpy()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recs', nargs='*', default=DEFAULT_RECS,
                        help='待分析的推荐文件列表')
    args = parser.parse_args()

    train_df, user_to_idx, item_to_idx = load_train_split(TRAIN_SPLIT_PATH)
    mat = build_matrix(train_df, user_to_idx, item_to_idx)
    pop = np.zeros(len(item_to_idx))
    for item_id, cnt in train_df['item_id'].value_counts().items():
        pop[item_to_idx[item_id]] = cnt
    total_pop = float(pop.sum())
    sim = compute_similarity(mat, 'cosine')

    rows = []
    for path in args.recs:
        if not os.path.exists(path):
            print(f"跳过（文件不存在）: {path}")
            continue
        recs = pd.read_csv(path)
        div = compute_diversity(recs, pop, total_pop, sim=sim)
        rows.append({
            '推荐文件': os.path.basename(path),
            '覆盖率': div['coverage'],
            '推荐图书数': div['distinct_items'],
            '平均热门度分位': div['avg_pop_pct'],
            '新颖度': div['novelty'],
            'ILD': div['ild'],
            '头部图书占比': head_share(recs, pop),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    print("=" * 70)
    print("推荐多样性指标对比（固定切分 Seed 42）")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")
    print("解读：覆盖率与新颖度越高、ILD 与平均热门度分位越低，说明推荐越多样、越不依赖热门书；")
    print("      F1 与多样性通常此消彼长，最终取舍以官方指标为优先。")
    print(f"\n结果已保存: {OUT_PATH}")


if __name__ == '__main__':
    main()
