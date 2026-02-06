"""
Ensemble Classifier for ML Model
================================

This module defines the EnsembleClassifier used in training and inference.
It must be importable by both training scripts and the inference code.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble of classifiers with weighted averaging.
    
    Combines predictions from multiple models (e.g., HistGB, RF, LR) to reduce
    variance and improve robustness across different market conditions.
    
    The predictions are averaged with weights, which provides implicit
    probability calibration through model averaging.
    """
    
    def __init__(self, models, weights=None, scaler=None):
        """
        Args:
            models: List of fitted classifiers
            weights: List of weights for each model (must sum to ~1)
            scaler: StandardScaler for models that need scaled input
        """
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        self.scaler = scaler
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y):
        """Ensemble is already fitted externally."""
        return self
    
    def predict_proba(self, X):
        """Average probability predictions from all models."""
        X_arr = np.array(X)
        weighted_probas = np.zeros((len(X_arr), 2))
        
        for model, weight in zip(self.models, self.weights):
            # Check if this model needs scaled input
            if hasattr(model, '_needs_scaling') and model._needs_scaling and self.scaler:
                X_input = self.scaler.transform(X_arr)
            else:
                X_input = X_arr
            
            proba = model.predict_proba(X_input)
            weighted_probas += proba * weight
        
        return weighted_probas
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    @property
    def n_iter_(self):
        """Return n_iter_ from first model (HistGB) for metadata compatibility."""
        if hasattr(self.models[0], 'n_iter_'):
            return self.models[0].n_iter_
        return None
    
    def __reduce__(self):
        """Custom pickle support - ensures class can be found during unpickling."""
        return (
            self.__class__,
            (self.models, self.weights, self.scaler)
        )
