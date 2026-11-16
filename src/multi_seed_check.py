"""
成员 B 补充实验：多随机种子稳定性复测

方案要求「至少对重点候选进行稳定性复测」，避免只对单次切分过拟合。
做法：复用成员 A 的切分函数（make_split），仅更换随机种子生成若干备用切分，
在同一批备用切分上分别评估热门度基线、BPR 之外的重点候选（ItemKNN 最优配置），
记录均值与波动，确认排名是否稳定。

注意：正式切分仍是 A 的 seed=42（outputs/train_split.csv），本脚本只做稳定性检验，
不改动正式切分与正式实验记录。

输出：outputs/multi_seed_results.csv, outputs/multi_seed_report.txt

用法（在项目根目录执行）：
    python src/multi_seed_check.py
    python src/multi_seed_check.py --seeds 43,44,45
"""

import argparse
import os

import numpy as np
import pandas as pd

from evaluate import evaluate
from item_knn import (
    build_matrix,
    compute_similarity,
    recommend_all,
    truncate_neighbors,
)
from make_split import make_split

TRAIN_PATH = os.path.join('data', 'train_dataset.csv')
RESULT_PATH = os.path.join('outputs', 'multi_seed_results.csv')
REPORT_PATH = os.path.join('outputs', 'multi_seed_report.txt')
TOP_K = 10

# 重点候选配置：正式实验 B-001 与网格搜索最优各一份，用于判断微弱提升是否为噪声
CONFIGS = [
    dict(name='ItemKNN(cosine,N=100,beta=0.5)', mode='cosine', n=100, beta=0.5,
         alpha=0.0, shrinkage=0.0, user_weight=0.0, hist_norm='sum'),
    dict(name='ItemKNN(cosine,N=50,beta=0.5,gamma=0.5)', mode='cosine', n=50, beta=0.5,
         alpha=0.0, shrinkage=0.0, user_weight=0.5, hist_norm='sum'),
]


def build_maps(train_df):
    users = np.sort(train_df['user_id'].unique())
    items = np.sort(train_df['item_id'].unique())
    user_to_idx = {int(u): i for i, u in enumerate(users)}
    item_to_idx = {int(i): j for j, i in enumerate(items)}
    return user_to_idx, item_to_idx


def pop_baseline(user_train_items, popularity, user_to_idx, item_to_idx, top_k=TOP_K):
    n_items = len(item_to_idx)
    items_arr = np.array(sorted(item_to_idx.keys()))
    pop = np.zeros(n_items)
    for item_id, cnt in popularity.items():
        pop[item_to_idx[item_id]] = cnt
    order = np.argsort(pop)[::-1]
    recs = {}
    for user_id in user_to_idx:
        seen = user_train_items.get(user_id, set())
        out = []
        for j in order:
            item_id = int(items_arr[j])
            if item_id not in seen:
                out.append(item_id)
                if len(out) == top_k:
                    break
        recs[user_id] = out
    return recs


def run_seed(train_df, val_df, seed):
    user_to_idx, item_to_idx = build_maps(train_df)
    mat = build_matrix(train_df, user_to_idx, item_to_idx)
    user_train_items = {
        int(u): set(v.astype(int))
        for u, v in train_df.groupby('user_id')['item_id']
    }
    val_targets = {
        int(u): set(v.astype(int))
        for u, v in val_df.groupby('user_id')['item_id']
    }
    popularity = train_df['item_id'].value_counts().to_dict()

    # --- 热门度基线
    recs_pop = pop_baseline(user_train_items, popularity, user_to_idx, item_to_idx)
    res_pop = evaluate(recs_pop, val_targets, top_k=TOP_K,
                       user_train_items=user_train_items)

    # --- ItemKNN 重点候选（逐个配置评估）
    rows = []
    sim_cache = {}
    nb_cache = {}
    for cfg in CONFIGS:
        skey = (cfg['mode'], cfg['shrinkage'], cfg['user_weight'])
        if skey not in sim_cache:
            sim_cache[skey] = compute_similarity(
                mat, cfg['mode'], shrinkage=cfg['shrinkage'],
                user_weight=cfg['user_weight'])
        nkey = (skey, cfg['n'])
        if nkey not in nb_cache:
            nb_cache[nkey] = truncate_neighbors(sim_cache[skey], cfg['n'])

        recs_knn, _ = recommend_all(
            nb_cache[nkey], user_train_items, popularity, user_to_idx, item_to_idx,
            beta=cfg['beta'], alpha=cfg['alpha'], top_k=TOP_K,
            hist_norm=cfg['hist_norm'])

        # 用与主模型一致的回退规则补足到 Top-K（未读热门书）
        pop_order = sorted(item_to_idx, key=lambda i: -popularity.get(i, 0))
        for u in val_targets:
            seen = user_train_items.get(u, set())
            out = [x for x in recs_knn.get(u, []) if x not in seen]
            for item_id in pop_order:
                if len(out) >= TOP_K:
                    break
                if item_id not in seen and item_id not in out:
                    out.append(item_id)
            recs_knn[u] = out[:TOP_K]

        res = evaluate(recs_knn, val_targets, top_k=TOP_K,
                       user_train_items=user_train_items)
        print(f"    {cfg['name']}: micro-F1={res['micro_f1']:.6f}")
        rows.append({'seed': seed, '模型': cfg['name'], 'micro_F1': res['micro_f1'],
                     'macro_F1': res['macro_f1'], 'micro_P': res['micro_precision'],
                     'micro_R': res['micro_recall']})

    out = [{'seed': seed, '模型': '热门度 Top-10', 'micro_F1': res_pop['micro_f1'],
            'macro_F1': res_pop['macro_f1'], 'micro_P': res_pop['micro_precision'],
            'micro_R': res_pop['micro_recall']}]
    out.extend(rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='43,44',
                        help='备用随机种子列表，逗号分隔（42 为正式切分，不在此处重跑）')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    lines = []

    def say(text=''):
        print(text)
        lines.append(str(text))

    say("=" * 60)
    say("ItemKNN 多随机种子稳定性复测（成员 B）")
    say("=" * 60)
    say(f"正式切分：Seed 42（outputs/train_split.csv），本脚本不修改")
    say(f"备用种子：{seeds}（同一 make_split 函数，仅更换种子）")
    say("")

    full_train = pd.read_csv(TRAIN_PATH)
    rows = []
    for seed in seeds:
        print(f"\n---- Seed {seed} ----")
        train_split, val_split = make_split(full_train, seed=seed)
        print(f"训练 {len(train_split)} 行 / 验证 {len(val_split)} 行")
        rows.extend(run_seed(train_split, val_split, seed))

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')

    say("各切分结果：")
    say(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    say("")

    summary = df.groupby('模型')['micro_F1'].agg(['mean', 'std', 'min', 'max'])
    say("micro-F1@10 汇总（跨切分）：")
    say(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    say("")

    # 正式切分的结果作为参照写入对比
    say("说明：Seed 42 的正式结果为 热门度 0.065518 / ItemKNN 0.190614；")
    say("      若备用切分上 ItemKNN 仍稳定领先热门度基线且排名未反转，视为通过稳定性复测。")
    say("")
    say("成员 B：多随机种子复测完成。")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n报告已保存: {REPORT_PATH}")


if __name__ == '__main__':
    main()
