import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from src.models.calibration_layer import ProbabilityCalibrator

class MultimodalCalibratedModel:
    """
    O3 Technical Contribution: Multimodal Model with Probability Calibration (Platt / Isotonic).
    Combines Cognitive, Demographic, Behavioral/Lifestyle, and Clinical modalities.
    Fits probability calibrators on calibration partition to achieve low ECE <= 0.05.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        self.feature_names = []
        self.platt_calibrator = ProbabilityCalibrator(method="platt")
        self.isotonic_calibrator = ProbabilityCalibrator(method="isotonic")
        self.is_calibrated = False

    def fit(self, train_df: pd.DataFrame, calib_df: pd.DataFrame, feature_cols: list, target_col: str = "dementia_risk"):
        self.feature_names = feature_cols
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        # 1. Fit Multimodal GBDT
        self.model.fit(X_train, y_train)

        # 2. Fit Calibrators on Calibration partition
        X_calib = calib_df[feature_cols]
        y_calib = calib_df[target_col].values
        uncalib_probs_calib = self.model.predict_proba(X_calib)[:, 1]

        self.platt_calibrator.fit(uncalib_probs_calib, y_calib)
        self.isotonic_calibrator.fit(uncalib_probs_calib, y_calib)
        self.is_calibrated = True

    def predict_uncalibrated(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, df: pd.DataFrame, method: str = "isotonic") -> np.ndarray:
        uncalib_probs = self.predict_uncalibrated(df)
        if not self.is_calibrated:
            return uncalib_probs

        if method.lower() == "platt":
            return self.platt_calibrator.calibrate(uncalib_probs)
        elif method.lower() == "isotonic":
            return self.isotonic_calibrator.calibrate(uncalib_probs)
        else:
            return uncalib_probs

    def evaluate(self, test_df: pd.DataFrame, target_col: str = "dementia_risk") -> Dict[str, Dict[str, float]]:
        y_true = test_df[target_col].values
        
        uncalib_p = self.predict_uncalibrated(test_df)
        platt_p = self.predict_proba(test_df, method="platt")
        iso_p = self.predict_proba(test_df, method="isotonic")

        results = {}
        for name, probs in [("uncalibrated", uncalib_p), ("platt", platt_p), ("isotonic", iso_p)]:
            auroc = roc_auc_score(y_true, probs)
            brier = brier_score_loss(y_true, probs)
            ece = ProbabilityCalibrator.calculate_ece(y_true, probs, n_bins=10)
            results[name] = {
                "auroc": float(auroc),
                "ece": float(ece),
                "brier_score": float(brier)
            }
        return results

if __name__ == "__main__":
    from src.data.dataset_loader import MHASDatasetLoader
    loader = MHASDatasetLoader()
    splits = loader.get_multimodal_splits()

    mm_model = MultimodalCalibratedModel()
    mm_model.fit(splits["train"], splits["calib"], splits["all_features"])

    metrics = mm_model.evaluate(splits["test"])
    print("O3 Multimodal Calibrated Model Evaluation:")
    for variant, vals in metrics.items():
        print(f"  [{variant.upper()}] AUROC: {vals['auroc']:.4f} | ECE: {vals['ece']:.4f} | Brier: {vals['brier_score']:.4f}")
