"""
成员 A 任务 1-2：数据字段检查、空值/重复检查、用户与图书 ID 范围、数据统计
输出：数据概览统计结果（控制台打印 + 保存到 outputs/data_statistics.txt）
"""

import pandas as pd
import os
from pandas.api.types import is_integer_dtype

TRAIN_PATH = os.path.join('data', 'train_dataset.csv')
TEST_PATH = os.path.join('data', 'test_dataset.csv')
SUBMISSION_PATH = os.path.join('data', 'submission.csv')
OUTPUT_DIR = 'outputs'
STATS_PATH = os.path.join(OUTPUT_DIR, 'data_statistics.txt')


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    return train, test, submission


def check_fields(train, test, submission):
    print("=" * 60)
    print("1. 字段检查")
    print("=" * 60)
    print(f"train_dataset.csv 列名: {list(train.columns)}")
    print(f"test_dataset.csv  列名: {list(test.columns)}")
    print(f"submission.csv      列名: {list(submission.columns)}")
    expected = (
        list(train.columns) == ['user_id', 'item_id']
        and list(test.columns) == ['user_id']
        and list(submission.columns) == ['user_id', 'item_id']
    )
    print(f"字段格式检查: {'通过' if expected else '失败（需核对字段名和顺序）'}")
    print()
    return expected


def check_nulls(train, test, submission):
    print("=" * 60)
    print("2. 空值检查")
    print("=" * 60)
    print(f"train 空值数:\n{train.isnull().sum()}")
    print(f"test  空值数:\n{test.isnull().sum()}")
    print(f"sub   空值数:\n{submission.isnull().sum()}")
    no_nulls = not (train.isnull().any().any() or test.isnull().any().any() or submission.isnull().any().any())
    print(f"空值检查: {'通过' if no_nulls else '失败'}")
    print()
    return no_nulls


def check_duplicates(train):
    print("=" * 60)
    print("3. 重复交互检查（user_id, item_id）")
    print("=" * 60)
    dup_count = train.duplicated(subset=['user_id', 'item_id']).sum()
    print(f"重复的用户-图书交互对数量: {dup_count}")
    print()
    return int(dup_count)


def check_id_types(train, test, submission):
    print("=" * 60)
    print("4. ID 数据类型检查")
    print("=" * 60)
    columns = [
        ("train.user_id", train['user_id']),
        ("train.item_id", train['item_id']),
        ("test.user_id", test['user_id']),
        ("submission.user_id", submission['user_id']),
        ("submission.item_id", submission['item_id']),
    ]
    valid = True
    for name, values in columns:
        is_valid = is_integer_dtype(values.dtype) and (values >= 0).all()
        print(f"{name}: dtype={values.dtype}, {'通过' if is_valid else '失败（应为非负整数）'}")
        valid = valid and is_valid
    print()
    return valid


def check_id_ranges(train, test, submission):
    print("=" * 60)
    print("5. 用户与图书 ID 范围")
    print("=" * 60)
    print(f"train user_id: min={train['user_id'].min()}, max={train['user_id'].max()}, 唯一用户数={train['user_id'].nunique()}")
    print(f"train item_id: min={train['item_id'].min()}, max={train['item_id'].max()}, 唯一图书数={train['item_id'].nunique()}")
    print(f"test  user_id: min={test['user_id'].min()}, max={test['user_id'].max()}, 唯一用户数={test['user_id'].nunique()}")
    print(f"sub   item_id: min={submission['item_id'].min()}, max={submission['item_id'].max()}, 唯一图书数={submission['item_id'].nunique()}")
    print()


def check_test_coverage(train, test):
    print("=" * 60)
    print("6. 测试用户覆盖率（测试用户是否全部在训练集中）")
    print("=" * 60)
    train_users = set(train['user_id'].unique())
    test_users = set(test['user_id'].unique())
    covered = test_users & train_users
    not_covered = test_users - train_users
    print(f"训练集唯一用户数: {len(train_users)}")
    print(f"测试集唯一用户数: {len(test_users)}")
    print(f"在训练集中出现的测试用户数: {len(covered)}")
    print(f"不在训练集中的测试用户数: {len(not_covered)}")
    if len(not_covered) > 0:
        print(f"示例未覆盖用户: {sorted(list(not_covered))[:10]}")
    print()
    return len(not_covered)


def check_submission_sample(test, submission):
    print("=" * 60)
    print("7. 提交样例检查")
    print("=" * 60)
    row_counts = submission.groupby('user_id').size()
    test_users = set(test['user_id'])
    submission_users = set(submission['user_id'])
    missing_users = test_users - submission_users
    extra_users = submission_users - test_users
    wrong_count = row_counts[row_counts != 10]
    duplicate_pairs = int(submission.duplicated(['user_id', 'item_id']).sum())
    print(f"样例总行数: {len(submission)}")
    print(f"样例用户数: {len(submission_users)}")
    print(f"每位用户恰好 10 行的用户数: {int((row_counts == 10).sum())}")
    print(f"缺失测试用户数: {len(missing_users)}")
    print(f"多余用户数: {len(extra_users)}")
    print(f"用户内重复交互/推荐对数量: {duplicate_pairs}")
    print(f"唯一推荐图书数: {submission['item_id'].nunique()}")
    print(f"行数不等于 10 的用户数: {len(wrong_count)}")
    if submission['item_id'].nunique() == 1 and len(submission) > 0:
        print("警告：样例中所有行都使用同一个图书 ID，疑似占位样例。")
    print()
    return {
        'sample_rows': len(submission),
        'sample_users': len(submission_users),
        'sample_missing_users': len(missing_users),
        'sample_extra_users': len(extra_users),
        'sample_wrong_count_users': len(wrong_count),
        'sample_duplicate_pairs': duplicate_pairs,
        'sample_unique_items': int(submission['item_id'].nunique()),
    }


def compute_statistics(train):
    print("=" * 60)
    print("8. 数据统计概览")
    print("=" * 60)
    total_interactions = len(train)
    n_users = train['user_id'].nunique()
    n_items = train['item_id'].nunique()
    possible_pairs = n_users * n_items
    density = total_interactions / possible_pairs

    user_interaction_counts = train['user_id'].value_counts()
    item_interaction_counts = train['item_id'].value_counts()

    stats = {
        '交互总数': total_interactions,
        '唯一用户数': n_users,
        '唯一图书数': n_items,
        '用户-图书可能组合数': possible_pairs,
        '交互密度': density,
        '用户交互量_中位数': user_interaction_counts.median(),
        '用户交互量_最小': user_interaction_counts.min(),
        '用户交互量_最大': user_interaction_counts.max(),
        '用户交互量_均值': user_interaction_counts.mean(),
        '图书交互量_中位数': item_interaction_counts.median(),
        '图书交互量_最小': item_interaction_counts.min(),
        '图书交互量_最大': item_interaction_counts.max(),
        '图书交互量_均值': item_interaction_counts.mean(),
    }

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print()
    print("用户交互量分位数:")
    print(user_interaction_counts.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print()
    print("图书交互量分位数:")
    print(item_interaction_counts.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print()

    return stats


def save_statistics(stats, train, test, audit):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append("图书推荐系统 - 数据概览统计报告（成员 A）")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. 文件结构")
    lines.append(f"   train_dataset.csv: user_id, item_id")
    lines.append(f"   test_dataset.csv:  user_id")
    lines.append(f"   submission.csv:    user_id, item_id")
    lines.append("")
    lines.append("2. 基础统计")
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"   {k}: {v:.6f}")
        else:
            lines.append(f"   {k}: {v}")
    lines.append("")
    lines.append(f"3. 字段格式检查: {'通过' if audit['fields'] else '失败'}")
    lines.append(f"4. ID 类型检查: {'通过' if audit['id_types'] else '失败'}")
    lines.append(f"5. 空值检查: {'通过' if audit['no_nulls'] else '失败'}")
    lines.append(f"6. 重复交互对数量: {audit['duplicate_pairs']}")
    lines.append(
        f"7. 测试用户未覆盖数量: {audit['uncovered_users']} / {test['user_id'].nunique()}"
    )
    lines.append("8. 提交样例检查")
    lines.append(f"   总行数/用户数: {audit['sample_rows']} / {audit['sample_users']}")
    lines.append(f"   缺失/多余用户: {audit['sample_missing_users']} / {audit['sample_extra_users']}")
    lines.append(f"   非 10 行用户数: {audit['sample_wrong_count_users']}")
    lines.append(f"   用户内重复对数: {audit['sample_duplicate_pairs']}")
    lines.append(f"   唯一推荐图书数: {audit['sample_unique_items']}")
    lines.append("")
    lines.append("9. 用户交互量分位数")
    lines.append(train['user_id'].value_counts().describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_string())
    lines.append("")
    lines.append("10. 图书交互量分位数")
    lines.append(train['item_id'].value_counts().describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_string())
    lines.append("")
    lines.append(
        "11. 结论: 统计结果来自本次脚本实际检查；请结合上述异常数和覆盖率判断数据是否可用于建模。"
    )

    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"统计报告已保存至: {STATS_PATH}")


def main():
    train, test, submission = load_data()
    fields_ok = check_fields(train, test, submission)
    no_nulls = check_nulls(train, test, submission)
    duplicate_pairs = check_duplicates(train)
    id_types_ok = check_id_types(train, test, submission)
    check_id_ranges(train, test, submission)
    uncovered_users = check_test_coverage(train, test)
    submission_sample = check_submission_sample(test, submission)
    stats = compute_statistics(train)
    audit = {
        'fields': fields_ok,
        'no_nulls': no_nulls,
        'duplicate_pairs': duplicate_pairs,
        'id_types': id_types_ok,
        'uncovered_users': uncovered_users,
        **submission_sample,
    }
    save_statistics(stats, train, test, audit)
    audit_passed = fields_ok and no_nulls and id_types_ok and duplicate_pairs == 0
    print(f"成员 A 任务 1-2 {'完成' if audit_passed else '发现问题'}：数据统计已输出。")
    if not audit_passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()