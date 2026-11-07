"""Models and candidates. Change grids only before comparing validation results."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from common.workflow import LABELS, make_preprocessor


class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=64, epochs=80, learning_rate=0.001, seed=42):
        self.hidden, self.epochs, self.learning_rate, self.seed = hidden, epochs, learning_rate, seed

    def fit(self, X, y):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError('Install PyTorch to run the MLP: pip install torch') from exc
        torch.manual_seed(self.seed)
        self.classes_ = np.array(LABELS)
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        targets = torch.tensor([LABELS.index(v) for v in y], dtype=torch.long)
        self.model_ = nn.Sequential(nn.Linear(x.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, len(LABELS)))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model_(x), targets)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        import torch
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32)), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_model(name, X, seed):
    if name == 'dummy':
        return DummyClassifier(strategy='most_frequent'), {}
    prep = make_preprocessor(X)
    if name == 'logistic':
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        grid = {'model__C': [0.1, 1.0, 10.0], 'model__class_weight': [None, 'balanced']}
    elif name == 'random_forest':
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
        grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10], 'model__class_weight': [None, 'balanced']}
    elif name == 'mlp':
        clf = TorchMLP(seed=seed)
        grid = {}
    else:
        raise ValueError(f'Unknown model: {name}')
    return Pipeline([('preprocess', prep), ('model', clf)]), grid
