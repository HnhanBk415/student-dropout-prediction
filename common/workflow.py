"""Shared, reproducible data preparation and metrics for the project."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABELS = ['Dropout', 'Enrolled', 'Graduate']
TARGET = 'Target'
# UCI 697: integer codes are categorical even when stored as numbers.
CATEGORICAL = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
BINARY = ['Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']


def load_data(path):
    path = Path(path)
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if TARGET not in df or len(df.columns) < 2:
        raise ValueError('Expected UCI CSV separated by semicolons, with a Target column.')
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset(LABELS):
        raise ValueError('Target must contain only Dropout, Enrolled and Graduate, without missing values.')
    return df


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_splits(df, path, output, seed, test_size, validation_size):
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError('Test and validation proportions must be positive and sum to less than 1.')
    idx = np.arange(len(df))
    remain, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=df[TARGET])
    train, val = train_test_split(remain, test_size=validation_size/(1-test_size), random_state=seed, stratify=df.iloc[remain][TARGET])
    folds = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit, valid in cv.split(train, df.iloc[train][TARGET]):
        folds.append({'train': train[fit].tolist(), 'validation': train[valid].tolist()})
    result = {'sha256': fingerprint(path), 'n_rows': len(df), 'seed': seed, 'test_size': test_size,
              'validation_size': validation_size, 'train': train.tolist(), 'validation': val.tolist(),
              'test': test.tolist(), 'folds': folds}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_splits(path, data_path, df):
    s = json.loads(Path(path).read_text(encoding='utf-8'))
    if s['sha256'] != fingerprint(data_path) or s['n_rows'] != len(df):
        raise ValueError('Dataset differs from the dataset used to create splits. Do not reuse these indices.')
    parts = [set(s[k]) for k in ('train', 'validation', 'test')]
    if any(parts[i] & parts[j] for i in range(3) for j in range(i)) or set.union(*parts) != set(range(len(df))):
        raise ValueError('Invalid split indices.')
    return s


def feature_groups(df):
    cats = [c for c in df.columns if c in [x.strip() for x in CATEGORICAL] or c in BINARY or not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in cats and c != TARGET]
    return nums, cats


def make_preprocessor(df):
    nums, cats = feature_groups(df)
    # sparse=False works on scikit-learn <1.2, sparse_output on newer versions.
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    transformers = []
    if nums:
        transformers.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums))
    if cats:
        transformers.append(('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', encoder)]), cats))
    return ColumnTransformer(transformers)


def evaluate(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, labels=LABELS, average='macro', zero_division=0),
            'dropout_f2': fbeta_score(np.asarray(y) == 'Dropout', np.asarray(pred) == 'Dropout', beta=2, zero_division=0),
            'per_class': classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix_labels': LABELS, 'confusion_matrix': confusion_matrix(y, pred, labels=LABELS).tolist()}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: float(x)), encoding='utf-8')
