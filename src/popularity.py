"""
成员 A 任务 5：实现并复核热门度 Top-10 基线
策略：在训练切分中统计图书热度，按热度从高到低为每位用户推荐 10 本
      未在该用户训练部分中出现过的图书（已读过滤）。
输出：outputs/popularity_baseline_recs.csv, outputs/popularity_baseline_eval.txt
"""

import pandas as pd
import os
from evaluate import load_val_targets, evaluate, print_result, save_result

TRAIN_SPLIT_PATH = os.path.join('outputs', 'train_split.csv')
VAL_SPLIT_PATH = os.path.join('outputs', 'val_split.csv')
OUTPUT_DIR = 'outputs'
RECS_PATH = os.path.join(OUTPUT_DIR, 'popularity_baseline_recs.csv')
EVAL_PATH = os.path.join(OUTPUT_DIR, 'popularity_baseline_eval.txt')
TOP_K = 10


def build_popularity_model(train_split_df):
    """统计训练切分中每本图书的交互次数作为热度分数"""
    item_counts = train_split_df['item_id'].value_counts()
    return item_counts.to_dict()


def get_user_train_items(train_split_df):
    """每位用户在训练部分已交互的图书集合"""
    return {
        int(user_id): set(items.astype(int))
        for user_id, items in train_split_df.groupby('user_id')['item_id']
    }


def generate_recommendations(popularity_dict, user_train_items, all_users, top_k=TOP_K):
    """为每位用户生成 Top-K 热门度推荐，过滤已读图书"""
    # 按热度排序的图书列表
    sorted_items = sorted(popularity_dict.keys(), key=lambda x: popularity_dict[x], reverse=True)

    recommendations = {}
    fallback_count = 0
    for user_id in all_users:
        seen = user_train_items.get(user_id, set())
        recs = []
        for item_id in sorted_items:
            if item_id not in seen:
                recs.append(int(item_id))
            if len(recs) >= top_k:
                break
        # 若候选不足 top_k，用全局热门补足（不重复）
        if len(recs) < top_k:
            for item_id in sorted_items:
                if item_id not in seen and item_id not in recs:
                    recs.append(int(item_id))
                if len(recs) >= top_k:
                    break
            fallback_count += 1
        if len(recs) < top_k:
            raise ValueError(
                f"用户 {user_id} 仅有 {len(recs)} 本未交互候选，无法满足 Top-{top_k}"
            )
        recommendations[user_id] = recs

    print(f"热门度基线：{fallback_count} 位用户候选不足 {top_k} 本，已使用全局热门补足。")
    return recommendations


def save_recommendations(recommendations, path=RECS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for user_id, items in recommendations.items():
        for item_id in items:
            rows.append({'user_id': user_id, 'item_id': item_id})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"推荐结果已保存: {path} ({len(df)} 行)")


def main():
    train_split = pd.read_csv(TRAIN_SPLIT_PATH)
    val_targets = load_val_targets(VAL_SPLIT_PATH)

    # 1. 建立热门度模型
    popularity_dict = build_popularity_model(train_split)
    print(f"热门度模型：共 {len(popularity_dict)} 本图书有交互记录")

    # 2. 获取每位用户训练部分已交互图书（用于已读过滤）
    user_train_items = get_user_train_items(train_split)

    # 3. 所有测试/验证用户
    all_users = sorted(list(val_targets.keys()))

    # 4. 生成推荐
    recommendations = generate_recommendations(popularity_dict, user_train_items, all_users, top_k=TOP_K)

    # 5. 保存推荐
    save_recommendations(recommendations)

    # 6. 评估
    result = evaluate(
        recommendations,
        val_targets,
        top_k=TOP_K,
        user_train_items=user_train_items,
    )
    print_result(result)
    save_result(result, path=EVAL_PATH)

    print("成员 A 任务 5 完成：热门度 Top-10 基线已生成并评估。")


if __name__ == '__main__':
    main()