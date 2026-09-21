import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.evaluation.evaluate_all import evaluate_complete_system
from src.dashboard.app import app

def main():
    parser = argparse.ArgumentParser(description="HealthCare Data Analytics: Explainable Early-Risk Prediction for Alzheimer's and Related Dementias")
    parser.add_argument("--mode", type=str, default="all", choices=["evaluate", "dashboard", "all"],
                        help="Execution mode: 'evaluate' (run KPIs & tests), 'dashboard' (launch UI), or 'all'")
    args = parser.parse_args()

    if args.mode in ["evaluate", "all"]:
        print("\n=== STEP 1: RUNNING COMPLETE EVALUATION & NEGATIVE TESTS ===")
        evaluate_complete_system()

    if args.mode in ["dashboard", "all"]:
        print("\n=== STEP 2: LAUNCHING CLINICIAN DASHBOARD SERVER ===")
        print("Point your browser to: http://127.0.0.1:8050/\n")
        app.run(host="127.0.0.1", port=8050, debug=False)

if __name__ == "__main__":
    main()
