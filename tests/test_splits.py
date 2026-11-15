import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_split_functions():
    namespace = {}
    notebook = json.loads((ROOT / 'common/data_io.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(''.join(cell['source']), namespace)
    notebook = json.loads((ROOT / 'common/splits.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code'
                  and ''.join(cell['source']).startswith('def load_splits('))
    exec(source, namespace)
    return namespace


class LoadSplitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_split_functions()
        cls.data_path = ROOT / 'data/raw/data.csv'
        cls.frame = pd.read_csv(cls.data_path, sep=';')
        cls.original = json.loads((ROOT / 'data/splits/splits.json').read_text(encoding='utf-8'))

    def load(self, split):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            path.write_text(json.dumps(split), encoding='utf-8')
            return self.functions['load_splits'](path, self.data_path, self.frame)

    def test_existing_split_is_accepted_unchanged(self):
        self.assertEqual(self.load(self.original), self.original)

    def test_duplicates_in_each_partition_are_rejected(self):
        for name in ('train', 'validation', 'test'):
            with self.subTest(partition=name):
                split = copy.deepcopy(self.original)
                row = split[name][0]
                split[name].append(row)
                with self.assertRaisesRegex(ValueError, f'Duplicate row indices in {name}:') as caught:
                    self.load(split)
                self.assertIn(str(row), str(caught.exception))

    def test_duplicates_in_each_fold_list_are_rejected(self):
        for number in range(len(self.original['folds'])):
            for name in ('train', 'validation'):
                with self.subTest(fold=number + 1, partition=name):
                    split = copy.deepcopy(self.original)
                    indices = split['folds'][number][name]
                    row = indices[0]
                    indices.append(row)
                    with self.assertRaisesRegex(ValueError, f'Duplicate row indices in fold {number + 1} {name}:') as caught:
                        self.load(split)
                    self.assertIn(str(row), str(caught.exception))

    def test_existing_overlap_guard_still_rejects(self):
        split = copy.deepcopy(self.original)
        split['validation'].append(split['train'][0])
        with self.assertRaisesRegex(ValueError, 'Invalid split indices'):
            self.load(split)

    def test_existing_fingerprint_guard_still_rejects(self):
        split = copy.deepcopy(self.original)
        split['sha256'] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'Dataset differs'):
            self.load(split)


if __name__ == '__main__':
    unittest.main()
