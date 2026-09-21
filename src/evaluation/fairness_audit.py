import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix, roc_auc_score

class SubgroupFairnessAuditor:
    """
    Independent Acceptance & Fairness Audit System (AC-1, KPI-3).
    Evaluates Equalized Odds (TPR and FPR parity across race/sex/age subgroups)
    and calculates subgroup performance metrics.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate_subgroup_fairness(
        self, 
        df: pd.DataFrame, 
        y_prob: np.ndarray, 
        subgroup_col: str, 
        target_col: str = "dementia_risk"
    ) -> Dict[str, Any]:
        
        df_eval = df.copy()
        df_eval["y_prob"] = y_prob
        df_eval["y_pred"] = (y_prob >= self.threshold).astype(int)
        
        subgroups = df_eval[subgroup_col].dropna().unique()
        subgroup_metrics = {}

        tprs = []
        fprs = []
        aurocs = []

        for sg in subgroups:
            sub_df = df_eval[df_eval[subgroup_col] == sg]
            if len(sub_df) == 0:
                continue

            y_true = sub_df[target_col].values
            y_pred = sub_df["y_pred"].values
            y_p = sub_df["y_prob"].values

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            auroc = roc_auc_score(y_true, y_p) if len(np.unique(y_true)) > 1 else 0.5

            tprs.append(tpr)
            fprs.append(fpr)
            aurocs.append(auroc)

            subgroup_metrics[str(sg)] = {
                "n_samples": len(sub_df),
                "prevalence": float(y_true.mean()),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "precision": float(precision),
                "auroc": float(auroc)
            }

        # Equalized Odds Gap: max(abs(tpr_i - tpr_j)) + max(abs(fpr_i - fpr_j))
        tpr_gap = float(np.max(tprs) - np.min(tprs)) if len(tprs) > 1 else 0.0
        fpr_gap = float(np.max(fprs) - np.min(fprs)) if len(fprs) > 1 else 0.0
        equalized_odds_gap = tpr_gap + fpr_gap

        return {
            "subgroup_col": subgroup_col,
            "subgroups": subgroup_metrics,
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
            "equalized_odds_gap": equalized_odds_gap,
            "worst_subgroup_auroc": float(np.min(aurocs)) if len(aurocs) > 0 else 0.0
        }

if __name__ == "__main__":
    from src.data.dataset_loader import MHASDatasetLoader
    from src.models.multimodal_model import MultimodalCalibratedModel

    loader = MHASDatasetLoader()
    splits = loader.get_multimodal_splits()

    mm_model = MultimodalCalibratedModel()
    mm_model.fit(splits["train"], splits["calib"], splits["all_features"])

    test_probs = mm_model.predict_proba(splits["test"], method="isotonic")
    auditor = SubgroupFairnessAuditor()

    sex_audit = auditor.evaluate_subgroup_fairness(splits["test"], test_probs, "sex_label")
    print("Sex Subgroup Fairness Audit:")
    print(f"  Equalized Odds Gap: {sex_audit['equalized_odds_gap']:.4f} (TPR Gap: {sex_audit['tpr_gap']:.4f}, FPR Gap: {sex_audit['fpr_gap']:.4f})")
    for sg, vals in sex_audit["subgroups"].items():
        print(f"  [{sg}] AUROC: {vals['auroc']:.4f} | TPR: {vals['tpr']:.4f} | FPR: {vals['fpr']:.4f}")
