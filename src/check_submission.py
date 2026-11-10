"""
成员 A 任务 7：检查测试用户覆盖率、推荐数量、重复图书、已读过滤与最终 CSV 格式
用于最终提交文件的自动化质量检查
输出：outputs/submission_check_report.txt
"""

import pandas as pd
import os
import sys
from pandas.api.types import is_integer_dtype

TRAIN_PATH = os.path.join('data', 'train_dataset.csv')
TEST_PATH = os.path.join('data', 'test_dataset.csv')
SUBMISSION_PATH = os.path.join('data', 'submission_final.csv')
OUTPUT_DIR = 'outputs'
CHECK_REPORT_PATH = os.path.join(OUTPUT_DIR, 'submission_check_report.txt')

TOP_K = 10


def load_data(submission_path):
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(submission_path)
    return train, test, submission


def check_columns(submission):
    print("=" * 60)
    print("1. 列名检查")
    print("=" * 60)
    expected = ['user_id', 'item_id']
    actual = list(submission.columns)
    ok = actual == expected
    print(f"期望列名: {expected}")
    print(f"实际列名: {actual}")
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok


def check_user_coverage(test, submission):
    print("=" * 60)
    print("2. 用户集合覆盖检查")
    print("=" * 60)
    test_users = set(test['user_id'].unique())
    sub_users = set(submission['user_id'].unique())
    missing = test_users - sub_users
    extra = sub_users - test_users
    print(f"测试集用户数: {len(test_users)}")
    print(f"提交集用户数: {len(sub_users)}")
    print(f"缺失用户数: {len(missing)}")
    print(f"多余用户数: {len(extra)}")
    ok = len(missing) == 0 and len(extra) == 0
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok, missing, extra


def check_rows_per_user(submission):
    print("=" * 60)
    print("3. 每位用户推荐数量检查（应为 10 行）")
    print("=" * 60)
    counts = submission.groupby('user_id').size()
    not_10 = counts[counts != TOP_K]
    print(f"总用户数: {len(counts)}")
    print(f"推荐数 != {TOP_K} 的用户数: {len(not_10)}")
    if len(not_10) > 0:
        print("示例异常用户:")
        print(not_10.head(10))
    ok = len(not_10) == 0
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok, not_10


def check_duplicates_within_user(submission):
    print("=" * 60)
    print("4. 用户内重复图书检查（每位用户 10 本应互不重复）")
    print("=" * 60)
    dup_mask = submission.duplicated(subset=['user_id', 'item_id'], keep=False)
    dup_users = submission.loc[dup_mask, 'user_id'].unique()
    print(f"存在重复图书的用户数: {len(dup_users)}")
    if len(dup_users) > 0:
        print("示例异常用户:", sorted(dup_users)[:10])
    ok = len(dup_users) == 0
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok, dup_users
def check_valid_items(train, submission):
    print("=" * 60)
    print("5. 图书 ID 有效性检查（item_id 必须在训练集范围内）")
    print("=" * 60)
    train_items = set(train['item_id'].unique())
    sub_items = set(submission['item_id'].unique())
    invalid = sub_items - train_items
    print(f"训练集唯一图书数: {len(train_items)}")
    print(f"提交中无效图书数: {len(invalid)}")
    if len(invalid) > 0:
        print("示例无效 ID:", sorted(list(invalid))[:10])
    ok = len(invalid) == 0
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok, invalid


def check_already_read(train, submission):
    print("=" * 60)
    print("6. 已读过滤检查（不应推荐用户在训练集中已交互过的图书）")
    print("=" * 60)
    train_read = set(zip(train['user_id'], train['item_id']))
    violations = [
        (uid, iid)
        for uid, iid in zip(submission['user_id'], submission['item_id'])
        if (uid, iid) in train_read
    ]

    print(f"违反已读过滤的推荐数: {len(violations)}")
    if len(violations) > 0:
        print("示例违规:", violations[:10])
    ok = len(violations) == 0
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok, violations


def check_nulls_and_types(submission):
    print("=" * 60)
    print("7. 空值与数据类型检查")
    print("=" * 60)
    nulls = submission.isnull().sum().sum()
    print(f"空值总数: {nulls}")
    print(f"user_id 类型: {submission['user_id'].dtype}")
    print(f"item_id 类型: {submission['item_id'].dtype}")
    integer_types = (
        is_integer_dtype(submission['user_id'].dtype)
        and is_integer_dtype(submission['item_id'].dtype)
    )
    ok = nulls == 0 and integer_types
    print(f"整数类型检查: {'通过' if integer_types else 'FAIL'}")
    print(f"检查结果: {'通过' if ok else 'FAIL'}")
    print()
    return ok


def check_not_sample(submission):
    print("=" * 60)
    print("8. 原始样例检查（不应是 submission.csv 的重复占位结果）")
    print("=" * 60)
    item_counts = submission['item_id'].value_counts()
    if item_counts.empty:
        print("检查结果: FAIL（提交文件为空）")
        print()
        return False
    most_common_item = item_counts.index[0]
    most_common_count = item_counts.iloc[0]
    total_rows = len(submission)
    print(f"最常见图书 ID: {most_common_item}，出现次数: {most_common_count}/{total_rows}")
    if most_common_count == total_rows:
        print("警告：所有用户都推荐了同一本书，可能是原始样例占位！")
        ok = False
    else:
        print("检查结果：通过（推荐结果有差异化）")
        ok = True
    print()
    return ok
def save_report(results, path=CHECK_REPORT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append("图书推荐系统 - 最终提交文件质量检查报告（成员 A）")
    lines.append("=" * 60)
    lines.append("")
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        lines.append(f"[{status}] {name}")
        if detail:
            lines.append(f"       {detail}")
    lines.append("")
    all_pass = all(r[1] for r in results)
    lines.append("=" * 60)
    lines.append(f"整体结果: {'全部通过' if all_pass else '存在不合格项，需修正'}")
    lines.append("=" * 60)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"质量检查报告已保存: {path}")
    return all_pass


def main(submission_path=SUBMISSION_PATH):
    if not os.path.exists(submission_path):
        print(f"错误：提交文件不存在: {submission_path}")
        print("请先生成最终提交文件，或指定正确路径。")
        return False

    train, test, submission = load_data(submission_path)

    results = []

    ok = check_columns(submission)
    results.append(("列名格式", ok, ""))
    if not ok:
        save_report(results)
        return False

    ok, missing, extra = check_user_coverage(test, submission)
    results.append(("用户集合覆盖", ok, f"缺失{len(missing)}人，多余{len(extra)}人"))

    ok, not_10 = check_rows_per_user(submission)
    detail = f"{len(not_10)} 位用户推荐数 != {TOP_K}" if len(not_10) > 0 else ""
    results.append((f"每位用户 {TOP_K} 行", ok, detail))

    ok, dup_users = check_duplicates_within_user(submission)
    results.append(("用户内去重", ok, f"{len(dup_users)} 位用户存在重复推荐"))

    ok, invalid = check_valid_items(train, submission)
    results.append(("图书 ID 有效性", ok, f"{len(invalid)} 个无效 ID"))

    ok, violations = check_already_read(train, submission)
    results.append(("已读过滤", ok, f"{len(violations)} 条违规推荐"))

    ok = check_nulls_and_types(submission)
    results.append(("空值与类型", ok, ""))

    ok = check_not_sample(submission)
    results.append(("非原始样例", ok, ""))

    all_pass = save_report(results)
    print()
    if all_pass:
        print("成员 A 任务 7 完成：提交文件质量检查全部通过。")
    else:
        print("成员 A 任务 7 完成：检查发现问题，请参考报告修正后再提交。")
    return all_pass


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else SUBMISSION_PATH
    raise SystemExit(0 if main(path) else 1)