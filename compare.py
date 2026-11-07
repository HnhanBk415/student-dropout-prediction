"""Compare completed validation runs; never reads the held-out test set."""
import argparse
import json
import hashlib
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='artifacts')
    a = p.parse_args()
    root = Path(a.output)
    rows = []
    current_split = hashlib.sha256((root/'splits.json').read_bytes()).hexdigest()
    for path in sorted((root/'validation').glob('*/metrics.json')):
        metrics = json.loads(path.read_text(encoding='utf-8'))
        config = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        if config.get('split_file_sha256') != current_split:
            raise SystemExit(f'Stale validation results: {path.parent}. Re-run the model with the current splits.')
        rows.append({'model': config['model'], 'cv_macro_f1': config['cv_macro_f1'], **{k:metrics[k] for k in ('accuracy', 'macro_f1', 'dropout_f2')}})
    if not rows:
        raise SystemExit('No validation results found. Run run_model.py first.')
    result = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    result.to_csv(root/'validation_comparison.csv', index=False)
    print(result.to_string(index=False))
if __name__ == '__main__':
    main()
