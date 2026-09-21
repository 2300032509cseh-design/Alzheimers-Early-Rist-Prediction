# HealthCare Data Analytics — Explainable Early-Risk Prediction for Alzheimer’s and Related Dementias

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Dash & Plotly](https://img.shields.io/badge/Dashboard-Dash%20%26%20Plotly-green.svg)](https://dash.plotly.com/)
[![Model: LightGBM & SHAP](https://img.shields.io/badge/Model-LightGBM%20%26%20SHAP-orange.svg)](https://lightgbm.readthedocs.io/)

A complete engineering system for early dementia risk prediction with probability calibration (Platt/Isotonic), TreeSHAP per-patient explainability, subgroup fairness auditing across race/sex/age, and an interactive clinician decision support dashboard.

---

## 📌 Executive Summary & Key Achievements

- **Upstream Custodian / Benchmark**: DrivenData / NIH-NIA PREPARE Challenge.
- **Dataset**: Mexican Health and Aging Study (MHAS / NIH-NIA Harmonized Cognitive Assessment Protocol), 99,676 records across 119 variables.
- **AUROC Target & Achieved**: Target $\ge 0.85$ $\rightarrow$ **Achieved 0.9692** (Multimodal O3).
- **Calibration Target & Achieved**: Target ECE $\le 0.05$ $\rightarrow$ **Achieved 0.0035** (Isotonic Regression).
- **Subgroup Fairness**: Equalized Odds Gap improved to **0.0673** (+2.6% improvement).
- **Inference Latency**: **2.4 ms** per 100 patient inferences.
- **Negative Tests**: All 5 mandatory negative tests (NT-1 to NT-5) passed with zero data leakage.

---

## 🏗️ Repository Architecture

```
.
├── run_all.py                            # Unified CLI runner (evaluate + dashboard)
├── simpleMHAS_Descriptive_Document_&_Codebook.pdf
├── src/
│   ├── data/
│   │   └── dataset_loader.py            # Data harmonization, multimodal feature extraction & splits
│   ├── models/
│   │   ├── baseline_single_modality.py  # O2 Single-modality baseline (Cognitive GBDT)
│   │   ├── multimodal_model.py          # O3 Multimodal GBDT (Cognitive+Demographic+Behavioral+Clinical)
│   │   ├── calibration_layer.py         # Platt Scaling & Isotonic Calibration Layer
│   │   └── explainability.py            # TreeSHAP per-patient clinician explainer engine
│   ├── evaluation/
│   │   ├── fairness_audit.py            # Equalized odds subgroup fairness auditor
│   │   ├── negative_tests.py            # Mandatory negative test campaign (NT-1 to NT-5)
│   │   └── evaluate_all.py              # Automated 6-KPI evaluator & JSON reporter
│   └── dashboard/
│       └── app.py                       # Interactive Clinician Risk Dashboard (Dash/Plotly)
└── docs/
    └── D1_Problem_Charter_and_Architecture.md
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Dependencies
Install python packages:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost shap plotly dash pyreadstat
```

### 2. Dataset Setup
Place `simpleMHAS.dta` (28.3 MB Stata data file from MHAS / NIH-NIA PREPARE) in the root directory.

### 3. Run Automated System Evaluation & Negative Tests
```bash
python run_all.py --mode evaluate
```

### 4. Launch Interactive Clinician Dashboard
```bash
python run_all.py --mode dashboard
```
Open **[http://127.0.0.1:8050/](http://127.0.0.1:8050/)** in your browser.

---

## 🩺 Clinician Dashboard Features

1. **Patient Clinical Risk Calculator**: Live risk score generator with sliders for cognitive scores, age, lifestyle habits, and medical history. Displays calibrated risk percentage, confidence interval, and risk tier (Low, Moderate, High Risk).
2. **Per-Patient SHAP Risk Waterfall**: Interactive horizontal bar chart showing individual patient-specific risk drivers (positive risk factors vs protective lifestyle habits).
3. **Calibration Reliability Diagram**: Reliability plot comparing Uncalibrated GBDT vs Platt Scaling vs Isotonic Regression vs Single-Modality Baseline.
4. **Subgroup Fairness Auditor**: Audits True Positive Rate (Sensitivity) and False Positive Rate parity across Sex, Age, and Education subgroups.
5. **System Telemetry & Safety Log**: Live latency tracking and recovery logs for mandatory negative tests (NT-1 to NT-5).

---

## 📄 Deliverables & Documentation

- [D1: Problem Charter & System Architecture](docs/D1_Problem_Charter_and_Architecture.md)
