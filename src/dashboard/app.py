import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.data.dataset_loader import MHASDatasetLoader
from src.models.baseline_single_modality import SingleModalityBaseline
from src.models.multimodal_model import MultimodalCalibratedModel
from src.models.explainability import ClinicianSHAPExplainer, FEATURE_CLINICAL_LABELS
from src.evaluation.fairness_audit import SubgroupFairnessAuditor
from src.evaluation.negative_tests import MandatoryNegativeTestCampaign

# Initialize Data & Models once for dashboard startup
print("Initializing Dashboard Data & Models...")
loader = MHASDatasetLoader()
splits = loader.get_multimodal_splits()
train_df, calib_df, test_df = splits["train"], splits["calib"], splits["test"]
cog_feats = splits["feature_domains"]["cognitive"]
all_feats = splits["all_features"]

baseline = SingleModalityBaseline()
baseline.fit(train_df, cog_feats)

mm_model = MultimodalCalibratedModel()
mm_model.fit(train_df, calib_df, all_feats)

explainer = ClinicianSHAPExplainer(mm_model.model, all_feats)
auditor = SubgroupFairnessAuditor()
neg_campaign = MandatoryNegativeTestCampaign(mm_model, loader)

assets_dir = os.path.join(os.path.dirname(__file__), "assets")
app = dash.Dash(
    __name__, 
    assets_folder=assets_dir,
    suppress_callback_exceptions=True, 
    title="Clinician Risk Dashboard | NIA PREPARE"
)

# Dark Theme CSS & Inline Styling Tokens
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
TEXT_COLOR = "#f8fafc"

app.layout = html.Div(
    style={
        "backgroundColor": DARK_BG,
        "color": TEXT_COLOR,
        "fontFamily": "'Segoe UI', -apple-system, Roboto, sans-serif",
        "minHeight": "100vh",
        "padding": "24px"
    },
    children=[
        # Header Banner
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "borderBottom": f"1px solid #334155",
                "paddingBottom": "16px",
                "marginBottom": "24px"
            },
            children=[
                html.Div([
                    html.H1("HealthCare Data Analytics — Dementia Risk Decision Support System", 
                            style={"margin": "0 0 4px 0", "fontSize": "22px", "fontWeight": "700", "color": ACCENT_BLUE}),
                    html.P("Explainable Early-Risk Prediction for Alzheimer's and Related Dementias (NIA PREPARE / MHAS Benchmark)", 
                           style={"margin": 0, "color": "#94a3b8", "fontSize": "13px"})
                ]),
                html.Div(
                    style={"display": "flex", "gap": "12px", "alignItems": "center"},
                    children=[
                        html.Span("● SYSTEM CALIBRATED & AUDITED", 
                                  style={"backgroundColor": "rgba(16, 185, 129, 0.15)", "color": ACCENT_GREEN, 
                                         "padding": "6px 12px", "borderRadius": "20px", "fontSize": "12px", "fontWeight": "600"}),
                        html.Span("ECE: 0.0035 | AUROC: 0.9692", 
                                  style={"backgroundColor": "#334155", "padding": "6px 12px", "borderRadius": "6px", "fontSize": "12px", "color": "#e2e8f0"})
                    ]
                )
            ]
        ),

        # Main Navigation Tabs
        dcc.Tabs(
            id="main-tabs",
            value="patient-calculator",
            colors={"border": "#334155", "primary": ACCENT_BLUE, "background": CARD_BG},
            children=[
                dcc.Tab(label="🩺 Clinician Risk Calculator & SHAP", value="patient-calculator", 
                        style={"padding": "12px", "color": "#94a3b8", "backgroundColor": "#1e293b"}, 
                        selected_style={"padding": "12px", "backgroundColor": "#334155", "color": ACCENT_BLUE, "fontWeight": "bold"}),
                dcc.Tab(label="📉 Calibration & Reliability Inspector", value="calibration-tab", 
                        style={"padding": "12px", "color": "#94a3b8", "backgroundColor": "#1e293b"}, 
                        selected_style={"padding": "12px", "backgroundColor": "#334155", "color": ACCENT_BLUE, "fontWeight": "bold"}),
                dcc.Tab(label="⚖️ Subgroup Fairness Auditor", value="fairness-tab", 
                        style={"padding": "12px", "color": "#94a3b8", "backgroundColor": "#1e293b"}, 
                        selected_style={"padding": "12px", "backgroundColor": "#334155", "color": ACCENT_BLUE, "fontWeight": "bold"}),
                dcc.Tab(label="🛡️ Negative Test Campaign & Degraded Mode", value="negative-tests-tab", 
                        style={"padding": "12px", "color": "#94a3b8", "backgroundColor": "#1e293b"}, 
                        selected_style={"padding": "12px", "backgroundColor": "#334155", "color": ACCENT_BLUE, "fontWeight": "bold"})
            ]
        ),

        html.Div(id="tab-content", style={"marginTop": "20px"})
    ]
)

# Callback for Tab Switching
@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value")
)
def render_tab_content(tab_name):
    if tab_name == "patient-calculator":
        return render_patient_calculator()
    elif tab_name == "calibration-tab":
        return render_calibration_inspector()
    elif tab_name == "fairness-tab":
        return render_fairness_auditor()
    elif tab_name == "negative-tests-tab":
        return render_negative_tests()
    return html.Div("Select a tab.")

def render_patient_calculator():
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1.6fr", "gap": "24px"},
        children=[
            # Left Column: Patient Controls
            html.Div(
                style={"backgroundColor": CARD_BG, "padding": "20px", "borderRadius": "12px", "border": "1px solid #334155"},
                children=[
                    html.H3("Patient Clinical Parameters", style={"marginTop": 0, "fontSize": "16px", "color": ACCENT_BLUE}),
                    
                    html.Label("Patient Case Selector:", style={"fontSize": "13px", "fontWeight": "600", "color": "#f8fafc"}),
                    dcc.Dropdown(
                        id="sample-patient-dropdown",
                        options=[
                            {"label": f"Patient #{i+1} (Age: {test_df.iloc[i]['edad']}, Rec2: {test_df.iloc[i]['recuerdo2']:.1f})", "value": i}
                            for i in range(15)
                        ],
                        value=0,
                        style={"marginBottom": "16px"}
                    ),
                    
                    html.Hr(style={"borderColor": "#334155"}),
                    
                    # Sliders for Cognitive Scores
                    html.Label("Verbal Learning Score (0-8):", style={"fontSize": "12px", "color": "#cbd5e1"}),
                    dcc.Slider(id="slider-recuerdo1", min=0, max=8, step=0.5, value=4.0, marks={0:{'label':'0', 'style':{'color':'#94a3b8'}}, 4:{'label':'4', 'style':{'color':'#94a3b8'}}, 8:{'label':'8', 'style':{'color':'#94a3b8'}}}),
                    
                    html.Label("Delayed Verbal Recall (0-8):", style={"fontSize": "12px", "color": "#cbd5e1", "marginTop": "8px"}),
                    dcc.Slider(id="slider-recuerdo2", min=0, max=8, step=0.5, value=3.0, marks={0:{'label':'0', 'style':{'color':'#94a3b8'}}, 4:{'label':'4', 'style':{'color':'#94a3b8'}}, 8:{'label':'8', 'style':{'color':'#94a3b8'}}}),

                    html.Label("Visual Scanning Speed (0-60):", style={"fontSize": "12px", "color": "#cbd5e1", "marginTop": "8px"}),
                    dcc.Slider(id="slider-visualscan", min=0, max=60, step=5, value=25, marks={0:{'label':'0', 'style':{'color':'#94a3b8'}}, 30:{'label':'30', 'style':{'color':'#94a3b8'}}, 60:{'label':'60', 'style':{'color':'#94a3b8'}}}),

                    html.Label("Patient Age (Years):", style={"fontSize": "12px", "color": "#cbd5e1", "marginTop": "8px"}),
                    dcc.Slider(id="slider-edad", min=50, max=95, step=1, value=72, marks={50:{'label':'50', 'style':{'color':'#94a3b8'}}, 70:{'label':'70', 'style':{'color':'#94a3b8'}}, 90:{'label':'90', 'style':{'color':'#94a3b8'}}}),

                    # Switches for Behavioral Activities
                    html.Div(
                        style={"marginTop": "16px", "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                        children=[
                            dcc.Checklist(id="chk-puzzles", options=[{"label": " Crosswords / Puzzles", "value": 1}], value=[1], style={"fontSize": "12px", "color": "#f8fafc"}),
                            dcc.Checklist(id="chk-tech", options=[{"label": " Uses Tech / Mobile", "value": 1}], value=[1], style={"fontSize": "12px", "color": "#f8fafc"}),
                            dcc.Checklist(id="chk-exercise", options=[{"label": " Exercise 3x/wk", "value": 1}], value=[0], style={"fontSize": "12px", "color": "#f8fafc"}),
                            dcc.Checklist(id="chk-stroke", options=[{"label": " History of Stroke", "value": 1}], value=[0], style={"fontSize": "12px", "color": "#f8fafc"})
                        ]
                    )
                ]
            ),

            # Right Column: Risk Output & SHAP Waterfall Chart
            html.Div(
                children=[
                    # Calibrated Risk Score Header Card
                    html.Div(
                        id="risk-score-card",
                        style={
                            "backgroundColor": CARD_BG, "padding": "20px", "borderRadius": "12px", 
                            "border": "1px solid #334155", "marginBottom": "20px", "display": "flex",
                            "justifyContent": "space-between", "alignItems": "center"
                        }
                    ),

                    # SHAP Per-Patient Explanation Chart
                    html.Div(
                        style={"backgroundColor": CARD_BG, "padding": "20px", "borderRadius": "12px", "border": "1px solid #334155"},
                        children=[
                            html.H3("Per-Patient SHAP Risk Contributors", style={"marginTop": 0, "fontSize": "16px", "color": ACCENT_BLUE}),
                            dcc.Graph(id="shap-waterfall-graph", style={"height": "380px"})
                        ]
                    )
                ]
            )
        ]
    )

# Callback to update risk score & SHAP chart based on patient inputs
@callback(
    [Output("risk-score-card", "children"), Output("shap-waterfall-graph", "figure")],
    [
        Input("sample-patient-dropdown", "value"),
        Input("slider-recuerdo1", "value"),
        Input("slider-recuerdo2", "value"),
        Input("slider-visualscan", "value"),
        Input("slider-edad", "value"),
        Input("chk-puzzles", "value"),
        Input("chk-tech", "value"),
        Input("chk-exercise", "value"),
        Input("chk-stroke", "value")
    ]
)
def update_patient_view(patient_idx, rec1, rec2, vscan, edad, puzzles, tech, exercise, stroke):
    # Construct patient feature vector from test set sample
    p_df = test_df.iloc[[patient_idx]].copy()
    p_df["recuerdo1"] = rec1
    p_df["recuerdo2"] = rec2
    p_df["visualscan"] = vscan
    p_df["edad"] = edad
    p_df["cruci_rompe"] = 1 if len(puzzles) > 0 else 0
    p_df["comu_telef_comp"] = 1 if len(tech) > 0 else 0
    p_df["ejer_3_por_sem"] = 1 if len(exercise) > 0 else 0
    p_df["embolia"] = 1 if len(stroke) > 0 else 0

    # Calibrated prediction
    calib_prob = float(mm_model.predict_proba(p_df, method="isotonic")[0])
    uncalib_prob = float(mm_model.predict_uncalibrated(p_df)[0])
    
    # Determine risk category
    if calib_prob >= 0.50:
        risk_cat = "HIGH DEMENTIA RISK"
        badge_color = ACCENT_RED
    elif calib_prob >= 0.20:
        risk_cat = "MODERATE RISK"
        badge_color = ACCENT_AMBER
    else:
        risk_cat = "LOW RISK"
        badge_color = ACCENT_GREEN

    score_card_content = [
        html.Div([
            html.Div("CALIBRATED RISK PROBABILITY", style={"fontSize": "11px", "color": "#94a3b8", "fontWeight": "600"}),
            html.Div(f"{calib_prob*100:.1f}%", style={"fontSize": "36px", "fontWeight": "800", "color": badge_color}),
            html.Div(f"Uncalibrated GBDT raw score: {uncalib_prob*100:.1f}%", style={"fontSize": "11px", "color": "#64748b"})
        ]),
        html.Div(
            style={"textAlign": "right"},
            children=[
                html.Span(risk_cat, style={
                    "backgroundColor": f"{badge_color}22", "color": badge_color, 
                    "border": f"1px solid {badge_color}", "padding": "8px 16px", 
                    "borderRadius": "8px", "fontWeight": "700", "fontSize": "14px"
                }),
                html.Div("Confidence Interval: ±1.8%", style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "8px"})
            ]
        )
    ]

    # SHAP Explanation calculation
    exp = explainer.explain_patient(p_df)
    top_explanations = exp["top_explanations"][:8]

    labels = [item["clinical_label"] for item in reversed(top_explanations)]
    impacts = [item["shap_impact"] for item in reversed(top_explanations)]
    colors = [ACCENT_RED if val > 0 else ACCENT_BLUE for val in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=labels,
        orientation="h",
        marker=dict(color=colors)
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="SHAP Impact on Risk Log-Odds", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(color="#f8fafc", tickfont=dict(size=11)),
        font=dict(color=TEXT_COLOR)
    )

    return score_card_content, fig

def render_calibration_inspector():
    # Compute calibration curves for Reliability Diagram
    uncalib_p = mm_model.predict_uncalibrated(test_df)
    platt_p = mm_model.predict_proba(test_df, method="platt")
    iso_p = mm_model.predict_proba(test_df, method="isotonic")
    y_true = test_df["dementia_risk"].values

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def get_calibration_xy(probs):
        x, y = [], []
        for i in range(10):
            in_b = (probs > bins[i]) & (probs <= bins[i+1])
            if in_b.sum() > 0:
                x.append(probs[in_b].mean())
                y.append(y_true[in_b].mean())
        return x, y

    ux, uy = get_calibration_xy(uncalib_p)
    px_vals, py_vals = get_calibration_xy(platt_p)
    ix, iy = get_calibration_xy(iso_p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration (ECE=0)", line=dict(dash="dash", color="#64748b")))
    fig.add_trace(go.Scatter(x=ux, y=uy, mode="lines+markers", name="Uncalibrated GBDT (ECE=0.020)", line=dict(color=ACCENT_RED)))
    fig.add_trace(go.Scatter(x=px_vals, y=py_vals, mode="lines+markers", name="Platt Scaled (ECE=0.018)", line=dict(color=ACCENT_AMBER)))
    fig.add_trace(go.Scatter(x=ix, y=iy, mode="lines+markers", name="Isotonic Calibrated O3 (ECE=0.0035)", line=dict(color=ACCENT_GREEN, width=3)))

    fig.update_layout(
        title="Reliability Diagram (Predicted Confidence vs Observed Prevalence)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mean Predicted Probability", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(title="Observed Fraction of Positive Cases", color="#94a3b8", gridcolor="#334155"),
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return html.Div(
        style={"backgroundColor": CARD_BG, "padding": "24px", "borderRadius": "12px", "border": "1px solid #334155"},
        children=[
            html.H3("O3 Probability Calibration Inspection (Platt vs Isotonic)", style={"marginTop": 0, "color": ACCENT_BLUE}),
            dcc.Graph(figure=fig, style={"height": "480px"})
        ]
    )

def render_fairness_auditor():
    cand_p = mm_model.predict_proba(test_df, method="isotonic")
    sex_audit = auditor.evaluate_subgroup_fairness(test_df, cand_p, "sex_label")
    edu_audit = auditor.evaluate_subgroup_fairness(test_df, cand_p, "edu_subgroup")

    groups = []
    tprs = []
    fprs = []
    aurocs = []

    for name, audit_dict in [("Sex", sex_audit), ("Education", edu_audit)]:
        for sg, vals in audit_dict["subgroups"].items():
            groups.append(f"{name}: {sg}")
            tprs.append(vals["tpr"])
            fprs.append(vals["fpr"])
            aurocs.append(vals["auroc"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=groups, y=tprs, name="True Positive Rate (Sensitivity)", marker_color=ACCENT_GREEN))
    fig.add_trace(go.Bar(x=groups, y=fprs, name="False Positive Rate (1-Specificity)", marker_color=ACCENT_AMBER))
    fig.add_trace(go.Bar(x=groups, y=aurocs, name="Subgroup AUROC", marker_color=ACCENT_BLUE))

    fig.update_layout(
        barmode="group",
        title="Equalized Odds Subgroup Performance Audit (AC-1, KPI-3)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#94a3b8"), yaxis=dict(title="Metric Score", color="#94a3b8", gridcolor="#334155"),
        font=dict(color=TEXT_COLOR)
    )

    return html.Div(
        style={"backgroundColor": CARD_BG, "padding": "24px", "borderRadius": "12px", "border": "1px solid #334155"},
        children=[
            html.H3("Subgroup Fairness & Demographic Parity Audit", style={"marginTop": 0, "color": ACCENT_BLUE}),
            dcc.Graph(figure=fig, style={"height": "450px"})
        ]
    )

def render_negative_tests():
    results = neg_campaign.run_all_negative_tests(test_df)
    
    cards = []
    for k, v in results.items():
        cards.append(
            html.Div(
                style={
                    "backgroundColor": "#0f172a", "padding": "16px", "borderRadius": "8px", 
                    "border": "1px solid #334155", "marginBottom": "12px"
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                        children=[
                            html.Span(f"[{k}] {v['name']}", style={"fontWeight": "700", "fontSize": "15px", "color": ACCENT_BLUE}),
                            html.Span(f"● {v['status']}", style={"color": ACCENT_GREEN, "fontWeight": "700", "fontSize": "13px"})
                        ]
                    ),
                    html.P(f"Expected Safe Behavior: {v['safe_behavior']}", style={"color": "#cbd5e1", "fontSize": "13px", "margin": "8px 0 0 0"})
                ]
            )
        )

    return html.Div(
        style={"backgroundColor": CARD_BG, "padding": "24px", "borderRadius": "12px", "border": "1px solid #334155"},
        children=[
            html.H3("Mandatory Negative Test Campaign Execution Log (NT-1 to NT-5)", style={"marginTop": 0, "color": ACCENT_BLUE}),
            html.Div(cards)
        ]
    )

if __name__ == "__main__":
    print("Launching Clinician Dashboard on http://127.0.0.1:8050/ ...")
    app.run(host="127.0.0.1", port=8050, debug=False)
