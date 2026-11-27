"""Create an additional per-user validation split without replacing A's seed-42 split.

Example: python src/make_extra_split.py --seed 43
"""

import argparse
import os

import pandas as pd

from make_split import make_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.seed == 42:
        parser.error('seed 42 is the primary split; run make_split.py for it')
    output_dir = os.path.join('outputs', f'seed{args.seed}')
    os.makedirs(output_dir, exist_ok=True)
    full = pd.read_csv('data/train_dataset.csv', dtype={'user_id': 'int32', 'item_id': 'int32'})
    train, validation = make_split(full, seed=args.seed)
    train.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    validation.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    print(f'Seed {args.seed}: train={len(train)}, validation={len(validation)}, output={output_dir}')


if __name__ == '__main__':
    main()
