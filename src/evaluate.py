"""
成员 A 任务 4：编写统一的 Top-K 推荐评估程序
输出：micro/macro Precision@10, Recall@10, F1@10
接口：evaluate(recommendations, val_targets, top_k=10, user_train_items=None) -> dict
  recommendations: dict {user_id: [item_id, ...]} 长度应为 10
  val_targets: dict {user_id: set([item_id, ...])} 验证目标集合
  user_train_items: 可选，提供时检查推荐是否包含训练历史中的图书
"""

import os
from collections import defaultdict

OUTPUT_DIR = 'outputs'
VAL_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'val_split.csv')
EVAL_RESULT_PATH = os.path.join(OUTPUT_DIR, 'evaluation_result.txt')

TOP_K = 10


def load_val_targets(val_split_path=VAL_SPLIT_PATH):
    """从验证切分文件加载每位用户的验证目标集合"""
    import pandas as pd
    df = pd.read_csv(val_split_path)
    targets = defaultdict(set)
    for _, row in df.iterrows():
        targets[int(row['user_id'])].add(int(row['item_id']))
    return dict(targets)


def evaluate(recommendations, val_targets, top_k=TOP_K, user_train_items=None):
    """
    计算 micro/macro Precision@K, Recall@K, F1@K
    recommendations: dict {user_id: list[item_id]} (已按推荐顺序排列，建议长度 = top_k)
    val_targets: dict {user_id: set[item_id]}
    """
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    validate_recommendations(
        recommendations,
        val_targets.keys(),
        top_k=top_k,
        user_train_items=user_train_items,
    )

    micro_tp = 0
    micro_pred = 0
    micro_actual = 0
    f1_list = []
    precision_list = []
    recall_list = []

    for user_id, target_set in val_targets.items():
        rec_list = recommendations.get(user_id, [])
        rec_set = set(rec_list)
        tp = len(rec_set & target_set)

        precision = tp / top_k if top_k > 0 else 0.0
        recall = tp / len(target_set) if len(target_set) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        micro_tp += tp
        micro_pred += top_k
        micro_actual += len(target_set)

        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

    n_users = len(val_targets)
    micro_precision = micro_tp / micro_pred if micro_pred > 0 else 0.0
    micro_recall = micro_tp / micro_actual if micro_actual > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    macro_precision = sum(precision_list) / n_users if n_users > 0 else 0.0
    macro_recall = sum(recall_list) / n_users if n_users > 0 else 0.0
    macro_f1 = sum(f1_list) / n_users if n_users > 0 else 0.0

    result = {
        'top_k': top_k,
        'n_users': n_users,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_tp': micro_tp,
        'micro_pred': micro_pred,
        'micro_actual': micro_actual,
    }
    return result


def validate_recommendations(
    recommendations,
    user_ids,
    top_k=TOP_K,
    user_train_items=None,
):
    """Raise ValueError when recommendations violate the shared Top-K contract."""
    expected_users = set(user_ids)
    actual_users = set(recommendations)
    missing_users = expected_users - actual_users
    extra_users = actual_users - expected_users
    errors = []

    if missing_users:
        errors.append(f"缺少用户: {sorted(missing_users)[:10]}")
    if extra_users:
        errors.append(f"包含非目标用户: {sorted(extra_users)[:10]}")

    for user_id in expected_users & actual_users:
        items = recommendations[user_id]
        if len(items) != top_k:
            errors.append(f"用户 {user_id} 推荐数为 {len(items)}，应为 {top_k}")
        if len(set(items)) != len(items):
            errors.append(f"用户 {user_id} 存在重复推荐")
        if user_train_items is not None:
            seen = user_train_items.get(user_id, set())
            overlap = set(items) & seen
            if overlap:
                errors.append(f"用户 {user_id} 推荐了已交互图书: {sorted(overlap)[:10]}")

    if errors:
        raise ValueError("推荐结果不符合评估约定：\n" + "\n".join(errors[:20]))


def print_result(result):
    print("=" * 50)
    print("统一评估结果（Top-K = {}）".format(result['top_k']))
    print("=" * 50)
    print(f"参与评估用户数: {result['n_users']}")
    print(f"micro Precision@{result['top_k']}: {result['micro_precision']:.6f}")
    print(f"micro Recall@{result['top_k']}:    {result['micro_recall']:.6f}")
    print(f"micro F1@{result['top_k']}:        {result['micro_f1']:.6f}")
    print(f"macro Precision@{result['top_k']}: {result['macro_precision']:.6f}")
    print(f"macro Recall@{result['top_k']}:    {result['macro_recall']:.6f}")
    print(f"macro F1@{result['top_k']}:        {result['macro_f1']:.6f}")
    print(f"micro TP={result['micro_tp']}, Pred={result['micro_pred']}, Actual={result['micro_actual']}")
    print("=" * 50)


def save_result(result, path=EVAL_RESULT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("=" * 50)
    lines.append("图书推荐系统 - 统一评估报告（成员 A）")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Top-K: {result['top_k']}")
    lines.append(f"参与评估用户数: {result['n_users']}")
    lines.append("")
    lines.append("Micro 指标（先汇总所有用户，再计算）:")
    lines.append(f"  Precision@{result['top_k']}: {result['micro_precision']:.6f}")
    lines.append(f"  Recall@{result['top_k']}:    {result['micro_recall']:.6f}")
    lines.append(f"  F1@{result['top_k']}:        {result['micro_f1']:.6f}")
    lines.append("")
    lines.append("Macro 指标（先计算每用户，再取平均）:")
    lines.append(f"  Precision@{result['top_k']}: {result['macro_precision']:.6f}")
    lines.append(f"  Recall@{result['top_k']}:    {result['macro_recall']:.6f}")
    lines.append(f"  F1@{result['top_k']}:        {result['macro_f1']:.6f}")
    lines.append("")
    lines.append(f"micro TP={result['micro_tp']}, Pred={result['micro_pred']}, Actual={result['micro_actual']}")
    lines.append("")
    lines.append("评估口径说明：")
    lines.append("- 每位用户固定推荐 10 本图书（Top-10）")
    lines.append("- 验证目标为该用户在验证切分中留出的交互图书")
    lines.append("- Precision = 命中数 / 10；Recall = 命中数 / 该用户验证目标数；F1 = 2*P*R/(P+R)")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"评估报告已保存至: {path}")


def main():
    targets = {1: {10, 11}, 2: {20}}
    cases = {
        "验证目标全命中": ({1: [10, 11], 2: [20, 21]}, 1.0),
        "零命中": ({1: [30, 31], 2: [32, 33]}, 0.0),
        "部分命中": ({1: [10, 30], 2: [32, 20]}, 2 / 3),
    }
    for name, (recommendations, expected_micro_recall) in cases.items():
        result = evaluate(recommendations, targets, top_k=2)
        if result["micro_recall"] != expected_micro_recall:
            raise AssertionError(f"{name}评估结果不符合预期: {result}")
        print(f"{name}样例通过: micro-F1={result['micro_f1']:.6f}")

    invalid_cases = [
        ({1: [10, 10], 2: [20, 21]}, "重复"),
        ({1: [10], 2: [20, 21]}, "推荐数"),
        ({1: [10, 11], 3: [20, 21]}, "缺少用户"),
    ]
    for recommendations, expected_error in invalid_cases:
        try:
            evaluate(recommendations, targets, top_k=2)
        except ValueError as error:
            if expected_error not in str(error):
                raise AssertionError(f"预期错误 {expected_error!r}，实际为 {error}") from error
        else:
            raise AssertionError(f"未检测到预期异常: {expected_error}")

    try:
        evaluate(
            {1: [10, 11], 2: [20, 21]},
            targets,
            top_k=2,
            user_train_items={1: {10}},
        )
    except ValueError as error:
        if "已交互" not in str(error):
            raise
    else:
        raise AssertionError("未检测到已交互图书")

    print("重复、缺少推荐、用户集合和已读过滤检查通过。")


if __name__ == '__main__':
    main()