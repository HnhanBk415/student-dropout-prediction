"""Explicitly evaluate a selected validation configuration on test once."""
import argparse
import json
import hashlib
from pathlib import Path
from common.workflow import TARGET, load_data, load_splits, save_json
from models import build_model
from run_model import save_predictions

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True, choices=['dummy', 'logistic', 'random_forest', 'mlp'])
    p.add_argument('--output', default='artifacts')
    p.add_argument('--selection-locked', action='store_true', help='Confirm the team locked the model and configuration')
    a = p.parse_args()
    if not a.selection_locked:
        p.error('Test is reserved for final evaluation. Select a model first, then use --selection-locked.')
    root = Path(a.output)
    final = root/'test'
    if final.exists():
        raise SystemExit('Test output already exists. Inspect it; do not repeatedly evaluate candidates on test.')
    df = load_data(a.data)
    split = load_splits(root/'splits.json', a.data, df)
    config_path = root/'validation'/a.model/'config.json'
    if not config_path.exists():
        raise SystemExit('Run validation for this model before locking the selection.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('split_file_sha256') != hashlib.sha256((root/'splits.json').read_bytes()).hexdigest():
        raise SystemExit('Validation configuration does not match current split seed.')
    X, y = df.drop(columns=TARGET), df[TARGET]
    model, _ = build_model(a.model, X, split['seed'])
    model.set_params(**config['best_params'])
    final_train = split['train'] + split['validation']
    model.fit(X.iloc[final_train], y.iloc[final_train])
    final.mkdir(parents=True)
    metrics = save_predictions(model, X, y, split['test'], final/'predictions.csv')
    save_json(final/'metrics.json', metrics)
    save_json(final/'locked_config.json', {'validation_config': config, 'dataset_sha256': split['sha256'], 'split_file_sha256': config['split_file_sha256'], 'refit': 'train + validation'})
    print(json.dumps({'model': a.model, 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'], 'dropout_f2': metrics['dropout_f2']}, indent=2))
if __name__ == '__main__':
    main()
