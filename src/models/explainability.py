import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

FEATURE_CLINICAL_LABELS = {
    "recuerdo1": "Verbal Learning Score (0-8)",
    "recuerdo2": "Delayed Verbal Recall Score (0-8)",
    "copiafiguras1": "Constructional Praxis Score (0-2)",
    "copiafiguras2": "Constructional Praxis Recall (0-2)",
    "visualscan": "Visual Scanning Speed (0-60)",
    "orientacion": "Orientation Score (0-3)",
    "serial7": "Series 7 Subtraction Score (0-5)",
    "edad": "Patient Age (Years)",
    "sexo": "Biological Sex (1=Male, 0=Female)",
    "educacion": "Education Completed (Years)",
    "urbano": "Urban Residence (1=Urban, 0=Rural)",
    "n_res": "Household Size",
    "n_hijos_vivos": "Number of Living Children",
    "cruci_rompe": "Engages in Crosswords / Puzzles",
    "lee": "Reads Books / Periodicals",
    "juegos_mesa": "Plays Tabletop / Board Games",
    "comu_telef_comp": "Uses Computer / Mobile Tech",
    "voluntario": "Participates in Volunteer Work",
    "asiste_club": "Attends Sports / Social Club",
    "ejer_3_por_sem": "Physical Exercise 3x/Week",
    "tabaco": "Currently Smokes Tobacco",
    "alcohol": "Consumes Alcohol",
    "hipertension": "History of Hypertension",
    "diabetes": "History of Diabetes",
    "enf_pulm": "Respiratory Disease",
    "artritis": "Arthritis Diagnosis",
    "infarto": "History of Heart Attack",
    "embolia": "History of Stroke",
    "cancer": "Cancer Diagnosis",
    "n_sint_depr": "CES-D Depressive Symptoms (0-9)",
    "n_abvd": "ADL Limitations Count (0-5)",
    "n_aivd": "IADL Limitations Count (0-4)",
    "n_mov": "Mobility Limitations Count (0-5)",
    "bmi_imp": "Body Mass Index (BMI)"
}

class ClinicianSHAPExplainer:
    """
    TreeSHAP Per-Patient Explanation Engine (FR-4).
    Computes exact local SHAP feature contributions for individual patient risk reports.
    """
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        # Initialize TreeSHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            self.expected_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            self.expected_value = float(ev)

    def explain_patient(self, patient_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates per-patient SHAP waterfall components for clinician view.
        """
        X = patient_df[self.feature_names]
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output / binary GBDT shap value structure
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray):
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                sv = shap_values[0]
            else:
                sv = shap_values
        else:
            sv = shap_values

        feature_val_dict = X.iloc[0].to_dict()
        explanations = []

        for f_name, s_val in zip(self.feature_names, sv):
            clinical_label = FEATURE_CLINICAL_LABELS.get(f_name, f_name)
            raw_val = feature_val_dict[f_name]
            explanations.append({
                "feature_code": f_name,
                "clinical_label": clinical_label,
                "patient_val": raw_val,
                "shap_impact": float(s_val),
                "direction": "Risk-Increasing" if s_val > 0 else "Protective/Risk-Decreasing"
            })

        # Sort by absolute SHAP impact magnitude
        explanations.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        return {
            "base_value": self.expected_value,
            "patient_features": feature_val_dict,
            "top_explanations": explanations
        }

if __name__ == "__main__":
    from src.data.dataset_loader import MHASDatasetLoader
    from src.models.multimodal_model import MultimodalCalibratedModel
    
    loader = MHASDatasetLoader()
    splits = loader.get_multimodal_splits()
    
    mm_model = MultimodalCalibratedModel()
    mm_model.fit(splits["train"], splits["calib"], splits["all_features"])
    
    explainer = ClinicianSHAPExplainer(mm_model.model, splits["all_features"])
    sample_patient = splits["test"].iloc[[0]]
    exp = explainer.explain_patient(sample_patient)
    print("SHAP Per-Patient Explanation sample:")
    print(f"  Base Expected Log-Odds: {exp['base_value']:.4f}")
    for item in exp["top_explanations"][:5]:
        print(f"  - {item['clinical_label']} (val={item['patient_val']}): SHAP={item['shap_impact']:+.4f} ({item['direction']})")
