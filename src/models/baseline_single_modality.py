import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Dict, Any
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

class SingleModalityBaseline:
    """
    O2 Reference Baseline: Features and Single-Modality Model (Cognitive Domain Only).
    Serves as the benchmark reference against which the multimodal O3 contribution is evaluated.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=self.random_state,
            verbose=-1
        )
        self.feature_names = []

    def fit(self, train_df: pd.DataFrame, feature_cols: list, target_col: str = "dementia_risk"):
        self.feature_names = feature_cols
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        self.model.fit(X_train, y_train)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, test_df: pd.DataFrame, target_col: str = "dementia_risk") -> Dict[str, float]:
        y_true = test_df[target_col].values
        y_prob = self.predict_proba(test_df)

        auroc = roc_auc_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        loss = log_loss(y_true, y_prob)
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_prob, n_bins=10)

        return {
            "auroc": float(auroc),
            "ece": float(ece),
            "brier_score": float(brier),
            "log_loss": float(loss)
        }

    @staticmethod
    def _calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return ece

if __name__ == "__main__":
    from src.data.dataset_loader import MHASDatasetLoader
    loader = MHASDatasetLoader()
    splits = loader.get_multimodal_splits()

    baseline = SingleModalityBaseline()
    cog_features = splits["feature_domains"]["cognitive"]
    baseline.fit(splits["train"], cog_features)

    metrics = baseline.evaluate(splits["test"])
    print("O2 Single-Modality Baseline Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
