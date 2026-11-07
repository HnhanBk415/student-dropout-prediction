"""Create shared splits and training-only EDA. Run from the project root."""
import argparse
from pathlib import Path
import pandas as pd
from common.workflow import TARGET, BINARY, create_splits, feature_groups, load_data, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--seed', type=int, required=True, help='Seed agreed by the group')
    parser.add_argument('--test-size', type=float, required=True, help='Fraction of all rows')
    parser.add_argument('--validation-size', type=float, required=True, help='Fraction of all rows')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--overwrite-splits', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    split_path = out / 'splits.json'
    if split_path.exists() and not args.overwrite_splits:
        raise FileExistsError(f'{split_path} already exists; use --overwrite-splits only if the group agrees.')
    df = load_data(args.data)
    split = create_splits(df, args.data, split_path, args.seed, args.test_size, args.validation_size)
    train = df.iloc[split['train']]
    nums, cats = feature_groups(train.drop(columns=TARGET))
    eda = out / 'eda_train'
    eda.mkdir(parents=True, exist_ok=True)
    train.isna().sum().rename('missing_count').to_csv(eda / 'missing.csv')
    train[TARGET].value_counts().rename('count').to_csv(eda / 'target_counts.csv')
    train.describe(include='all').transpose().to_csv(eda / 'summary.csv')
    pd.crosstab(train[TARGET], columns='count', normalize='columns').to_csv(eda / 'class_proportions.csv')
    save_json(eda / 'overview.json', {'full_shape': list(df.shape), 'full_duplicate_rows': int(df.duplicated().sum()),
              'train_shape': list(train.shape), 'train_duplicate_rows': int(train.duplicated().sum()),
              'numeric_columns': nums, 'categorical_columns': [x for x in cats if x not in BINARY],
              'binary_columns': [x for x in cats if x in BINARY]})
    print(f'Splits and training EDA saved in {out}. Train/validation/test: '
          f"{len(split['train'])}/{len(split['validation'])}/{len(split['test'])}")

if __name__ == '__main__':
    main()
