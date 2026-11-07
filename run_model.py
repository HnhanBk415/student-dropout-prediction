"""Train on shared Train and evaluate on Validation; tune Logistic/RF on shared folds."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from common.workflow import LABELS, TARGET, evaluate, load_data, load_splits, save_json
from models import build_model


def fit_model(name, df, split):
    train_idx = split['train']
    X, y = df.drop(columns=TARGET), df[TARGET]
    model, grid = build_model(name, X, split['seed'])
    info = {'model': name, 'seed': split['seed'], 'best_params': {}, 'cv_macro_f1': None}
    if grid:
        positions = {row: i for i, row in enumerate(train_idx)}
        folds = [(np.array([positions[x] for x in f['train']]), np.array([positions[x] for x in f['validation']])) for f in split['folds']]
        search = GridSearchCV(model, grid, scoring='f1_macro', cv=folds, n_jobs=1, refit=True, error_score='raise')
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        model = search.best_estimator_
        info.update(best_params=search.best_params_, cv_macro_f1=float(search.best_score_))
    else:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
    return model, info


def save_predictions(model, X, y, indices, output):
    pred = model.predict(X.iloc[indices])
    proba = model.predict_proba(X.iloc[indices])
    result = pd.DataFrame({'row_index': indices, 'actual': y.iloc[indices].to_numpy(), 'predicted': pred})
    for label in LABELS:
        col = list(model.classes_).index(label) if label in model.classes_ else None
        result[f'prob_{label}'] = proba[:, col] if col is not None else 0.0
    result.to_csv(output, index=False)
    return evaluate(y.iloc[indices], pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True, choices=['dummy', 'logistic', 'random_forest', 'mlp'])
    p.add_argument('--output', default='artifacts')
    a = p.parse_args()
    out = Path(a.output)
    df = load_data(a.data)
    split = load_splits(out/'splits.json', a.data, df)
    model, info = fit_model(a.model, df, split)
    info['split_file_sha256'] = hashlib.sha256((out/'splits.json').read_bytes()).hexdigest()
    destination = out/'validation'/a.model
    destination.mkdir(parents=True, exist_ok=True)
    metrics = save_predictions(model, df.drop(columns=TARGET), df[TARGET], split['validation'], destination/'predictions.csv')
    save_json(destination/'metrics.json', metrics)
    save_json(destination/'config.json', info)
    print(json.dumps({'model': a.model, 'best_params': info['best_params'], 'cv_macro_f1': info['cv_macro_f1'],
                      'validation_accuracy': metrics['accuracy'], 'validation_macro_f1': metrics['macro_f1'],
                      'validation_dropout_f2': metrics['dropout_f2']}, indent=2))

if __name__ == '__main__':
    main()
