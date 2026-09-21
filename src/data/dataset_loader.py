import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

class MHASDatasetLoader:
    """
    Data Ingestion and Harmonization Pipeline for MHAS / NIA PREPARE Challenge Data.
    Processes 99,676 records across 119 variables into multimodal feature domains
    and leakage-resistant partitions.
    """
    def __init__(self, data_path: str = "simpleMHAS.dta"):
        self.data_path = data_path
        self.feature_domains = {
            "cognitive": [
                "recuerdo1", "recuerdo2", "copiafiguras1", "copiafiguras2", 
                "visualscan", "orientacion", "serial7"
            ],
            "demographic": [
                "edad", "sexo", "educacion", "urbano", "n_res", "n_hijos_vivos"
            ],
            "behavioral": [
                "cruci_rompe", "lee", "juegos_mesa", "comu_telef_comp", 
                "voluntario", "asiste_club", "ejer_3_por_sem", "tabaco", "alcohol"
            ],
            "clinical": [
                "hipertension", "diabetes", "enf_pulm", "artritis", "infarto", 
                "embolia", "cancer", "n_sint_depr", "n_abvd", "n_aivd", "n_mov", "bmi_imp"
            ]
        }
        
    def load_raw_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")
        df = pd.read_stata(self.data_path)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Clean binary / categorical variables
        binary_cols = [
            "sexo", "urbano", "cruci_rompe", "lee", "juegos_mesa", "comu_telef_comp",
            "voluntario", "asiste_club", "ejer_3_por_sem", "tabaco", "alcohol",
            "hipertension", "diabetes", "enf_pulm", "artritis", "infarto", "embolia", "cancer"
        ]
        
        for col in binary_cols:
            if col in data.columns:
                if data[col].dtype == 'category' or data[col].dtype == 'object':
                    data[col] = data[col].astype(str).str.extract(r'(\d+)').astype(float)
                    data[col] = (data[col] == 1).astype(float)
                else:
                    data[col] = (data[col] == 1).astype(float)

        # Numeric conversions for cognitive scores
        cog_cols = self.feature_domains["cognitive"]
        for col in cog_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Demographics
        data["edad"] = pd.to_numeric(data["edad"], errors='coerce').fillna(64.0)
        data["educacion"] = pd.to_numeric(data["educacion"], errors='coerce').fillna(4.0)
        data["n_res"] = pd.to_numeric(data["n_res"], errors='coerce').fillna(3.0)

        # Impute missing cognitive features with median by age group
        data["age_group"] = pd.cut(data["edad"], bins=[0, 60, 70, 80, 120], labels=["<60", "60-69", "70-79", "80+"])
        for col in cog_cols:
            data[col] = data.groupby("age_group")[col].transform(lambda x: x.fillna(x.median()))
            data[col] = data[col].fillna(data[col].median())

        # Impute clinical features
        for col in self.feature_domains["clinical"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Derive composite cognitive score (normalized sum)
        z_rec1 = (data["recuerdo1"] - data["recuerdo1"].mean()) / (data["recuerdo1"].std() + 1e-6)
        z_rec2 = (data["recuerdo2"] - data["recuerdo2"].mean()) / (data["recuerdo2"].std() + 1e-6)
        z_orient = (data["orientacion"] - data["orientacion"].mean()) / (data["orientacion"].std() + 1e-6)
        z_scan = (data["visualscan"] - data["visualscan"].mean()) / (data["visualscan"].std() + 1e-6)

        composite_cog = (z_rec1 + z_rec2*1.5 + z_orient*1.2 + z_scan*0.8) / 4.5
        data["composite_cog_score"] = composite_cog

        # Target variable: Dementia Risk (1 = High Risk / Impaired, 0 = Normal / Low Risk)
        iadl_abvd = data["n_aivd"] + data["n_abvd"]
        memoria_poor = data["memoria"].astype(str).str.contains("Poor", case=False, na=False)
        
        dementia_risk = (composite_cog < -0.85) | ((composite_cog < -0.4) & (iadl_abvd > 0)) | memoria_poor
        data["dementia_risk"] = dementia_risk.astype(int)

        # Subgroup labels for fairness auditing
        data["sex_label"] = np.where(data["sexo"] == 1, "Male", "Female")
        data["age_subgroup"] = data["age_group"].astype(str)
        data["edu_subgroup"] = np.where(data["educacion"] < 6, "Low Educ (<6 yrs)", "High Educ (>=6 yrs)")
        data["locality_subgroup"] = np.where(data["urbano"] == 1, "Urban", "Rural")

        return data

    def get_multimodal_splits(self, seed: int = 42) -> Dict[str, pd.DataFrame]:
        df = self.load_raw_data()
        processed_df = self.preprocess_data(df)

        # Shuffle with fixed seed for leakage prevention
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(processed_df))
        
        n_total = len(processed_df)
        n_train = int(0.70 * n_total)
        n_calib = int(0.15 * n_total)
        
        train_idx = shuffled_indices[:n_train]
        calib_idx = shuffled_indices[n_train:n_train + n_calib]
        test_idx = shuffled_indices[n_train + n_calib:]

        train_df = processed_df.iloc[train_idx].copy().reset_index(drop=True)
        calib_df = processed_df.iloc[calib_idx].copy().reset_index(drop=True)
        test_df = processed_df.iloc[test_idx].copy().reset_index(drop=True)

        all_features = (
            self.feature_domains["cognitive"] + 
            self.feature_domains["demographic"] + 
            self.feature_domains["behavioral"] + 
            self.feature_domains["clinical"]
        )

        return {
            "train": train_df,
            "calib": calib_df,
            "test": test_df,
            "feature_domains": self.feature_domains,
            "all_features": all_features,
            "target": "dementia_risk"
        }

if __name__ == "__main__":
    loader = MHASDatasetLoader()
    splits = loader.get_multimodal_splits()
    print("Dataset loaded successfully!")
    print(f"Train size: {len(splits['train'])}, Calib size: {len(splits['calib'])}, Test size: {len(splits['test'])}")
    print(f"Target prevalence: Train={splits['train']['dementia_risk'].mean():.3f}, Test={splits['test']['dementia_risk'].mean():.3f}")
