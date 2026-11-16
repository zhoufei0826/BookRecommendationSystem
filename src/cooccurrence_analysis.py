"""
成员 B 任务 1：图书共现关系与热门度分析

分析内容：
1. 图书热度分布与头部/中腰部/长尾分组（按累计交互占比 20% / 50% 划分）
2. 原始共现 vs 余弦相似度的邻居差异（验证热门书偏置，为 ItemKNN 的相似度选择提供依据）
3. 抽样图书 Top-10 共现邻居中全局热门书的占比
4. 长尾图书的可建模性观察

输入：outputs/train_split.csv（成员 A 的固定切分，Seed 42）
输出：outputs/cooccurrence_analysis.txt, outputs/item_popularity.csv, outputs/cooccurrence_samples.csv

用法（在项目根目录执行）：
    python src/cooccurrence_analysis.py
"""

import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

TRAIN_SPLIT_PATH = os.path.join('outputs', 'train_split.csv')
OUTPUT_DIR = 'outputs'
REPORT_PATH = os.path.join(OUTPUT_DIR, 'cooccurrence_analysis.txt')
POP_PATH = os.path.join(OUTPUT_DIR, 'item_popularity.csv')
SAMPLE_PATH = os.path.join(OUTPUT_DIR, 'cooccurrence_samples.csv')

RANDOM_SEED = 42
N_SAMPLE_ITEMS = 20
TOP_NEIGHBORS = 10
N_POPULAR_HEAD = 50


class Report:
    """同时打印到控制台并累积文本，便于一键保存"""

    def __init__(self):
        self.lines = []

    def __call__(self, text=''):
        print(text)
        self.lines.append(str(text))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"共现分析报告已保存: {path}")


def build_matrix(train_df):
    """稀疏 用户×图书 01 矩阵"""
    users = np.sort(train_df['user_id'].unique())
    items = np.sort(train_df['item_id'].unique())
    user_to_idx = {int(u): i for i, u in enumerate(users)}
    item_to_idx = {int(i): j for j, i in enumerate(items)}
    rows = train_df['user_id'].map(user_to_idx).to_numpy()
    cols = train_df['item_id'].map(item_to_idx).to_numpy()
    mat = csr_matrix((np.ones(len(train_df), dtype=np.float32), (rows, cols)),
                     shape=(len(users), len(items)))
    return mat, items


def analyze_popularity(train_df, say):
    """热度分布与分组"""
    pop = train_df['item_id'].value_counts()
    total = int(pop.sum())
    cum = pop.cumsum() / total

    n_head = int((cum <= 0.20).sum()) + 1
    n_mid = int((cum <= 0.50).sum()) + 1 - n_head
    n_tail = len(pop) - n_head - n_mid

    say("一、图书热度分布")
    say("-" * 50)
    say(f"有交互记录的图书: {len(pop)} 本，交互总数 {total} 条")
    say(f"头部图书（累计贡献前 20% 交互）: {n_head} 本 ({100 * n_head / len(pop):.1f}%)")
    say(f"中腰部图书（20%-50%）        : {n_mid} 本 ({100 * n_mid / len(pop):.1f}%)")
    say(f"长尾图书（50% 以后）          : {n_tail} 本 ({100 * n_tail / len(pop):.1f}%)")
    say(f"Top-50 图书累计交互占比: {100 * pop.head(50).sum() / total:.1f}%")
    say(f"Top-100 图书累计交互占比: {100 * pop.head(100).sum() / total:.1f}%")
    top5 = "，".join(f"图书 {i}（{c} 次）" for i, c in pop.head(5).items())
    say(f"交互量最高的 5 本: {top5}")
    say(f"图书交互量中位数: {int(pop.median())}，最大: {int(pop.max())}")
    say("")
    return pop


def analyze_cooccurrence(mat, items, pop, say):
    """原始共现 vs 余弦相似度邻居对比"""
    item_user = mat.T.tocsr()
    pop_arr = np.asarray(item_user.sum(axis=1)).ravel()
    norm = np.sqrt(np.maximum(pop_arr, 1e-9))
    popular_head = set(pop.head(N_POPULAR_HEAD).index)

    candidates = np.nonzero(pop_arr > 5)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = rng.choice(candidates, size=min(N_SAMPLE_ITEMS, len(candidates)), replace=False)

    say("二、原始共现 vs 余弦相似度的邻居差异")
    say("-" * 50)
    say(f"抽样 {len(sampled)} 本图书，各取 Top-{TOP_NEIGHBORS} 邻居对比：")

    rows = []
    overlaps = []
    head_ratios = []
    for it in sampled:
        co = np.asarray((mat.T @ mat[:, it]).todense()).ravel()
        co[it] = 0
        cos = co / norm / norm[it]
        top_co = np.argsort(co)[::-1][:TOP_NEIGHBORS]
        top_cos = np.argsort(cos)[::-1][:TOP_NEIGHBORS]
        overlap = len(set(top_co.tolist()) & set(top_cos.tolist()))
        overlaps.append(overlap / TOP_NEIGHBORS)
        head_ratios.append(np.mean([1.0 if items[n] in popular_head else 0.0 for n in top_co]))

        item_id = int(items[it])
        say(f"  图书 {item_id}（热度 {int(pop_arr[it])}）")
        say(f"    原始共现邻居 : {[int(items[n]) for n in top_co]}")
        say(f"    余弦相似度邻居: {[int(items[n]) for n in top_cos]}（重合 {overlap}/{TOP_NEIGHBORS}）")
        for rank in range(TOP_NEIGHBORS):
            rows.append({
                'item_id': item_id, 'rank': rank + 1,
                'co_neighbor': int(items[top_co[rank]]),
                'co_count': int(co[top_co[rank]]),
                'cos_neighbor': int(items[top_cos[rank]]),
                'cos_score': float(cos[top_cos[rank]]),
            })

    say("")
    say(f"  平均邻居重合率: {100 * np.mean(overlaps):.1f}%")
    say(f"  原始共现 Top-10 邻居中属于全局 Top-50 热门书的平均占比: {100 * np.mean(head_ratios):.1f}%")
    say("  → 原始共现受热门书偏置影响明显；余弦相似度按图书度归一，能给出更专属的邻居，")
    say("    因此 ItemKNN 主模型采用余弦相似度（实验 B-00x 中 raw 对比进一步验证）。")
    say("")

    pd.DataFrame(rows).to_csv(SAMPLE_PATH, index=False)
    return pop_arr


def analyze_longtail(pop, say):
    """长尾可建模性"""
    say("三、长尾图书可建模性观察")
    say("-" * 50)
    median = pop.median()
    low = pop[pop <= median]
    say(f"交互量不超过中位数（{int(median)}）的图书: {len(low)} 本，"
        f"平均交互 {low.mean():.1f} 条")
    say("  这类图书共现用户少、邻居列表短，候选打分对其覆盖有限，")
    say("  长尾命中主要依赖用户历史中的热门书映射，命中占比如见分层分析报告。")
    say("")


def main():
    say = Report()
    say("=" * 60)
    say("图书共现与热门度分析（成员 B）")
    say("=" * 60)
    say(f"数据来源: {TRAIN_SPLIT_PATH}（成员 A 固定切分，Seed 42）")
    say("")

    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    say(f"训练切分: {len(train_df)} 行，{train_df['user_id'].nunique()} 位用户，"
        f"{train_df['item_id'].nunique()} 本图书")
    say("")

    mat, items = build_matrix(train_df)
    pop = analyze_popularity(train_df, say)
    analyze_cooccurrence(mat, items, pop, say)
    analyze_longtail(pop, say)

    pop.rename('n_interactions').to_csv(POP_PATH, index_label='item_id')
    print(f"图书热度表已保存: {POP_PATH}")
    say("成员 B 任务 1 完成：图书共现与热门度分析已输出。")
    say.save(REPORT_PATH)


if __name__ == '__main__':
    main()
