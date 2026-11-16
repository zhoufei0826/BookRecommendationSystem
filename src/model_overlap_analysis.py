"""
成员 B 补充分析：ItemKNN（B-001）与 BPR（A-002）推荐结果重叠度分析
目的：为成员 C 的模型融合提供依据——两模型推荐是否互补、重叠是否过高
输出：outputs/model_overlap_analysis.txt
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

OUTPUT_DIR = 'outputs'
ITEMKNN_RECS = os.path.join(OUTPUT_DIR, 'itemknn_recs.csv')
BPR_RECS = os.path.join(OUTPUT_DIR, 'bpr_recs.csv')
POP_RECS = os.path.join(OUTPUT_DIR, 'popularity_baseline_recs.csv')
OUT_PATH = os.path.join(OUTPUT_DIR, 'model_overlap_analysis.txt')

TOP_K = 10


def load_recs(path):
    df = pd.read_csv(path)
    recs = defaultdict(list)
    for uid, iid in zip(df['user_id'], df['item_id']):
        recs[int(uid)].append(int(iid))
    return dict(recs)


def overlap_stats(recs_a, recs_b, label):
    """逐用户统计两个推荐列表的交集大小及分布"""
    common_users = set(recs_a) & set(recs_b)
    overlaps = np.array([
        len(set(recs_a[u]) & set(recs_b[u]))
        for u in sorted(common_users)
    ])
    union_sizes = np.array([
        len(set(recs_a[u]) | set(recs_b[u]))
        for u in sorted(common_users)
    ])
    return {
        'label': label,
        'n_users': len(common_users),
        'mean_overlap': overlaps.mean(),
        'median_overlap': float(np.median(overlaps)),
        'std_overlap': overlaps.std(),
        'overlap_0': float((overlaps == 0).mean()),
        'overlap_le2': float((overlaps <= 2).mean()),
        'overlap_ge5': float((overlaps >= 5).mean()),
        'overlap_10': float((overlaps == 10).mean()),
        'jaccard': float((overlaps / union_sizes).mean()),
        'overlaps': overlaps,
    }


def distinct_contribution(recs_a, recs_b, label_a, label_b):
    """B 的推荐中有多少书是 A 完全没推荐过的（B 对 A 的补充量）"""
    items_a = set()
    for v in recs_a.values():
        items_a.update(v)
    items_b = set()
    for v in recs_b.values():
        items_b.update(v)
    return {
        'label_a': label_a,
        'label_b': label_b,
        'items_a': len(items_a),
        'items_b': len(items_b),
        'items_common': len(items_a & items_b),
        'only_b': len(items_b - items_a),
        'only_b_ratio': len(items_b - items_a) / len(items_b),
    }


def main():
    print('加载推荐结果 ...')
    recs_knn = load_recs(ITEMKNN_RECS)
    recs_bpr = load_recs(BPR_RECS)
    recs_pop = load_recs(POP_RECS)

    print(f'ItemKNN 用户数 {len(recs_knn)}，BPR 用户数 {len(recs_bpr)}')

    lines = []
    lines.append('=' * 60)
    lines.append('成员 B 补充分析：ItemKNN 与 BPR 推荐重叠度（供成员 C 融合参考）')
    lines.append('=' * 60)
    lines.append('')

    pairs = [
        (recs_knn, recs_bpr, 'ItemKNN(B-001) vs BPR(A-002)'),
        (recs_knn, recs_pop, 'ItemKNN(B-001) vs 热门度基线'),
        (recs_bpr, recs_pop, 'BPR(A-002) vs 热门度基线'),
    ]

    lines.append('一、逐用户 Top-10 重叠（同一用户两个列表的交集大小，0~10）')
    lines.append('-' * 60)
    for a, b, label in pairs:
        s = overlap_stats(a, b, label)
        lines.append(f'{label}')
        lines.append(f"  平均重叠 {s['mean_overlap']:.2f} 本（中位数 {s['median_overlap']:.0f}，"
                     f"标准差 {s['std_overlap']:.2f}）")
        lines.append(f"  完全不重叠(0本)用户占比 {s['overlap_0']*100:.1f}%，"
                     f"≤2本 {s['overlap_le2']*100:.1f}%，"
                     f"≥5本 {s['overlap_ge5']*100:.1f}%，"
                     f"完全相同(10本) {s['overlap_10']*100:.1f}%")
        lines.append(f"  平均 Jaccard 相似度 {s['jaccard']:.3f}")
        lines.append('')

    lines.append('二、图书目录层面的互补性')
    lines.append('-' * 60)
    combos = [
        (recs_knn, recs_bpr, 'ItemKNN', 'BPR'),
        (recs_bpr, recs_knn, 'BPR', 'ItemKNN'),
    ]
    for a, b, la, lb in combos:
        d = distinct_contribution(a, b, la, lb)
        lines.append(f"{lb} 相对 {la}：{lb} 共覆盖 {d['items_b']} 本书，"
                     f"其中 {d['only_b']} 本（{d['only_b_ratio']*100:.1f}%）"
                     f"是 {la} 从未推荐过的")
    lines.append('')

    # 重叠数与用户活跃度的关系：重叠高的用户是否更活跃
    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_split.csv'),
                           usecols=['user_id'])
    activity = train_df['user_id'].value_counts()
    s = overlap_stats(recs_knn, recs_bpr, 'knn-bpr')
    overlaps = s.pop('overlaps')
    users = sorted(set(recs_knn) & set(recs_bpr))
    act = np.array([activity.get(u, 0) for u in users])
    hi = act >= np.median(act)
    lines.append('三、重叠与用户活跃度的关系（ItemKNN vs BPR）')
    lines.append('-' * 60)
    lines.append(f'高活跃半区（训练交互 ≥ 中位数 {int(np.median(act))}）：'
                 f'平均重叠 {overlaps[hi].mean():.2f} 本')
    lines.append(f'低活跃半区：平均重叠 {overlaps[~hi].mean():.2f} 本')
    lines.append(f'相关系数（活跃度 vs 重叠数）：{np.corrcoef(act, overlaps)[0, 1]:.3f}')
    lines.append('')

    # 融合可行性提示
    lines.append('四、对成员 C 融合的提示')
    lines.append('-' * 60)
    mean_ov = overlap_stats(recs_knn, recs_bpr, '')['mean_overlap']
    if mean_ov < 4:
        lines.append(f'- 两模型平均每用户仅重叠 {mean_ov:.1f}/10 本，互补性强，'
                     '融合（如加权排序合并、RRF）有明确的收益空间')
    else:
        lines.append(f'- 两模型平均每用户重叠 {mean_ov:.1f}/10 本，'
                     '互补性中等，融合收益需在验证集上实测')
    lines.append('- 融合候选集建议：并集去重后按模型归一化分数重排，'
                 '仅在验证集上选择融合权重（避免用验证目标调参泄漏）')
    lines.append('- 融合后同样必须满足：每人 10 本、无重复、不含训练已读书目')

    text = '\n'.join(lines)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'\n分析已保存至: {OUT_PATH}')


if __name__ == '__main__':
    main()
