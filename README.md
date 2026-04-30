# Mini Project: ML-Inspired Digital Verification Risk Scoring (Semiconductor)

This mini project demonstrates how ML-style risk scoring can prioritize digital verification effort in semiconductor design.

## Why this version?
This repository is designed to run in restricted environments, so the implementation uses only Python's standard library (no `pip install` required).

## What it does
- Generates synthetic verification-run records with realistic inputs:
  - block type
  - changed lines
  - toggling activity
  - lint warnings
  - CDC violations
  - timing slack
  - coverage delta
  - engineer experience
- Computes a logistic risk score for verification failure probability.
- Prints confusion-matrix style metrics and summary quality scores.
- Exports a ranked CSV report (`verification_risk_report.csv`) for triage.

## Step-by-step: Run the Project
1. **Open a terminal** and go to the project folder:
   ```bash
   cd /workspace/Mini-project-
   ```

2. **Check Python is available** (Python 3.9+ recommended):
   ```bash
   python3 --version
   ```

3. **Run the script**:
   ```bash
   python3 ml_digital_verification_semiconductor.py
   ```

4. **Review console output**:
   - Confusion matrix (TP/FP/FN/TN)
   - Accuracy, precision, recall, F1
   - Top 5 highest-risk verification runs

5. **Open generated report**:
   ```bash
   cat verification_risk_report.csv
   ```
   (or open it in Excel/Google Sheets for easier sorting/filtering)

## Quick Run
```bash
python3 ml_digital_verification_semiconductor.py
```

## Output
- Console metrics (accuracy, precision, recall, F1).
- Top highest-risk verification runs.
- CSV output sorted by risk for downstream review.
