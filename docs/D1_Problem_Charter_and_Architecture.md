# D1: Problem Charter, System Architecture, & Requirements Map

## 1. Executive Summary & Problem Charter
- **Project Title**: HealthCare Data Analytics — Explainable Early-Risk Prediction for Alzheimer’s and Related Dementias
- **Problem Custodian / Benchmark**: DrivenData / NIA PREPARE Challenge (U.S. National Institute on Aging / NIH-NIA)
- **Primary Beneficiaries**: Clinicians, Neurologists, and Clinical Reviewers.
- **Core Engineering Goal**: Deliver a calibrated, explainable dementia-risk prediction model with low Expected Calibration Error ($\text{ECE} \le 0.05$) and an audited, fairness-verified clinician dashboard.

## 2. System & Physical Architecture

```
+-----------------------------------------------------------------------------------+
|                            DATA INGESTION & HARMONIZATION                         |
|  MHAS / NIA PREPARE Harmonized Multimodal Dataset (simpleMHAS.dta - 99,676 Rec)   |
+-----------------------------------------------------------------------------------+
           |                          |                          |
           v                          v                          v
 +-------------------+      +-------------------+      +-------------------+
 | Cognitive Domain  |      | Demographic Domain|      | Behavioral & Tech |
 | Verbal Recall,    |      | Age, Sex, Edu,    |      | Crosswords, Tech, |
 | Praxis, Scan,     |      | Locality, Hhold   |      | Reading, Exercise |
 | Orientation       |      +-------------------+      +-------------------+
 +-------------------+                |                          |
           |                          +------------+-------------+
           v                                       v
 +--------------------------+            +-----------------------------------+
 |  O2 SINGLE-MODALITY      |            |  O3 MULTIMODAL GBDT MODEL         |
 |  BASELINE REFERENCE      |            |  LightGBM + Feature Fusion        |
 |  (Cognitive GBDT)        |            +-----------------------------------+
 +--------------------------+                              |
           |                                               v
           |                             +-----------------------------------+
           |                             |  O3 PROBABILITY CALIBRATION LAYER |
           |                             |  Platt Scaling / Isotonic Reg     |
           |                             |  Target ECE <= 0.05 (Achieved:   |
           |                             |  0.0035)                          |
           |                             +-----------------------------------+
           |                                               |
           +----------------------+------------------------+
                                  |
                                  v
 +----------------------------------------------------------------------------------+
 |                         O4 CLINICIAN DASHBOARD (Dash/Plotly)                    |
 | - Live Patient Risk Calculator                                                   |
 | - Per-Patient SHAP Risk Waterfall Chart                                         |
 | - Calibration & Reliability Inspector Diagram                                    |
 | - Subgroup Fairness Auditor (Equalized Odds)                                     |
 | - Degraded Mode & System Telemetry Monitor                                       |
 +----------------------------------------------------------------------------------+
                                  |
                                  v
 +----------------------------------------------------------------------------------+
 |                    O5 INDEPENDENT ACCEPTANCE & NEGATIVE TESTS                    |
 | - AC-1 to AC-4 Representative & Leakage-Resistant Partition Evaluation           |
 | - NT-1 to NT-5 Mandatory Negative Test Campaign Execution                        |
 +----------------------------------------------------------------------------------+
```

## 3. Requirements Traceability Matrix

| Requirement | Description | System Module | Status |
| :--- | :--- | :--- | :--- |
| **FR-1** | Ingest & validate NIA PREPARE / MHAS data | `src/data/dataset_loader.py` | VERIFIED |
| **FR-2** | Reproduce single-modality baseline reference (O2) | `src/models/baseline_single_modality.py` | VERIFIED |
| **FR-3** | Implement multimodal model with Platt/Isotonic calibration (O3) | `src/models/multimodal_model.py` | VERIFIED |
| **FR-4** | Implement TreeSHAP per-patient explanations | `src/models/explainability.py` | VERIFIED |
| **FR-5** | Implement interactive clinician risk dashboard (O4) | `src/dashboard/app.py` | VERIFIED |
| **FR-6** | Deliver complete demonstrable system | `run_all.py` | VERIFIED |
| **FR-7** | Execute NT-1 to NT-5 negative test campaign | `src/evaluation/negative_tests.py` | VERIFIED |
