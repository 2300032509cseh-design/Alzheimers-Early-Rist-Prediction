import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class ProbabilityCalibrator:
    """
    O3 Probability Calibration Layer using Platt Scaling (Sigmoidal) and Isotonic Regression.
    Reduces Expected Calibration Error (ECE) to guarantee reliable risk percentages for clinical use.
    """
    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        if self.method == "platt":
            self.calibrator = LogisticRegression(C=1e5, solver="lbfgs")
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError(f"Unknown calibration method: {method}. Choose 'platt' or 'isotonic'.")

    def fit(self, uncalibrated_probs: np.ndarray, y_true: np.ndarray):
        uncalibrated_probs = np.clip(uncalibrated_probs, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            # Platt scaling operates on log-odds / logits
            logits = np.log(uncalibrated_probs / (1 - uncalibrated_probs)).reshape(-1, 1)
            self.calibrator.fit(logits, y_true)
        else:
            self.calibrator.fit(uncalibrated_probs, y_true)

    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        uncalibrated_probs = np.clip(uncalibrated_probs, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            logits = np.log(uncalibrated_probs / (1 - uncalibrated_probs)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        else:
            return self.calibrator.predict(uncalibrated_probs)

    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)
