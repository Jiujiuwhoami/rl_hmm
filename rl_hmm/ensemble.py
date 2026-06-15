"""Ensemble classifier — combines multiple base models for collective prediction."""

from __future__ import annotations

import numpy as np


class EnsembleClassifier:
    """Custom ensemble that combines multiple pre-fitted base models.

    Two methods:
      - 'voting':   weighted average of base model probabilities
      - 'stacking': base model probabilities fed into a meta-model

    All base models must be fitted before being passed to this class.
    Supports .predict() and .predict_proba() for sklearn compatibility.
    """

    def __init__(self, estimators: list, method: str = 'voting',
                 weights: list[float] | None = None,
                 meta_model=None):
        """
        Args:
            estimators: list of (name, model) tuples, each pre-fitted
            method: 'voting' or 'stacking'
            weights: per-model weights for voting (None = equal)
            meta_model: fitted meta-model for stacking
        """
        self.estimators = estimators
        self.method = method
        self.weights = weights
        self.meta_model = meta_model
        self.n_classes_ = 2

    def predict_proba(self, X) -> np.ndarray:
        """Get class probabilities from the ensemble."""
        if self.method == 'voting':
            return self._voting_proba(X)
        return self._stacking_proba(X)

    def predict(self, X) -> np.ndarray:
        """Get class predictions (0/1)."""
        return self.predict_proba(X).argmax(axis=1)

    def _voting_proba(self, X) -> np.ndarray:
        """Weighted average of base model probabilities."""
        n_classes = self.n_classes_
        avg = np.zeros((X.shape[0], n_classes))
        total_weight = 0.0
        for i, (_, model) in enumerate(self.estimators):
            w = self.weights[i] if self.weights else 1.0
            proba = model.predict_proba(X)
            if proba.shape[1] == 1:
                proba = np.hstack([1 - proba, proba])
            avg += w * proba
            total_weight += w
        return avg / total_weight

    def _stacking_proba(self, X) -> np.ndarray:
        """Base model probabilities → meta-model prediction."""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def _get_meta_features(self, X) -> np.ndarray:
        """Collect probability predictions from all base models."""
        n = X.shape[0]
        n_est = len(self.estimators)
        meta = np.zeros((n, n_est))
        for i, (_, model) in enumerate(self.estimators):
            proba = model.predict_proba(X)
            meta[:, i] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return meta
