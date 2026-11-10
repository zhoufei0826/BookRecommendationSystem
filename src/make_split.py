"""
成员 A 任务 3：建立固定的训练/验证切分，并保存或确定性重建切分
输出：outputs/train_split.csv, outputs/val_split.csv, outputs/split_info.txt
策略：按用户分层随机留出约 20% 交互作为验证目标，其余用于训练。
      固定随机种子 42，保证可复现。
"""

import pandas as pd
import os
import json

TRAIN_PATH = os.path.join('data', 'train_dataset.csv')
OUTPUT_DIR = 'outputs'
TRAIN_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'train_split.csv')
VAL_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'val_split.csv')
SPLIT_INFO_PATH = os.path.join(OUTPUT_DIR, 'split_info.txt')

RANDOM_SEED = 42
VAL_RATIO = 0.20


def make_split(train_df, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """按用户分层切分：每位用户的交互中，约 val_ratio 作为验证目标"""
    train_list = []
    val_list = []

    grouped = train_df.groupby('user_id')
    for _, group in grouped:
        group = group.sample(frac=1.0, random_state=seed)  # 打乱该用户的交互顺序
        n_val = max(1, int(len(group) * val_ratio))  # 每位用户至少留 1 个验证样本
        val_list.append(group.iloc[:n_val])
        train_list.append(group.iloc[n_val:])

    train_split = pd.concat(train_list).reset_index(drop=True)
    val_split = pd.concat(val_list).reset_index(drop=True)
    return train_split, val_split


def save_split(train_split, val_split):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_split.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_split.to_csv(VAL_SPLIT_PATH, index=False)

    info = {
        'random_seed': RANDOM_SEED,
        'val_ratio': VAL_RATIO,
        'train_split_rows': len(train_split),
        'val_split_rows': len(val_split),
        'train_split_users': int(train_split['user_id'].nunique()),
        'val_split_users': int(val_split['user_id'].nunique()),
        'description': '按用户分层随机切分，固定随机种子 42，每位用户至少保留 1 个验证样本'
    }
    with open(SPLIT_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"训练切分已保存: {TRAIN_SPLIT_PATH} ({len(train_split)} 行)")
    print(f"验证切分已保存: {VAL_SPLIT_PATH} ({len(val_split)} 行)")
    print(f"切分信息已保存: {SPLIT_INFO_PATH}")
    return info


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"原始训练集: {len(train_df)} 行，{train_df['user_id'].nunique()} 位用户")
    train_split, val_split = make_split(train_df)
    info = save_split(train_split, val_split)
    print("成员 A 任务 3 完成：固定训练/验证切分已生成。")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()