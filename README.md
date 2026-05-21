# Loan Risk Prediction — Optimizing for Recall on Imbalanced Credit Data

A multi-class and binary classification pipeline that flags high-risk borrowers in a severely imbalanced lending dataset. Built around a deliberate tradeoff: **higher recall on bad loans at the cost of overall accuracy** — because in lending, a missed default is more expensive than a false alarm.

## Key Results

| Metric | Value |
|---|---|
| Overall accuracy | **89.5%** |
| Recall (bad-loan class) | **62%** |
| F1 (bad-loan class) | 0.28 |
| Model | LightGBM |
| Validation | Stratified K-Fold CV |

> **Why these numbers matter:** on a default-rate of ~[X]% (severely imbalanced), naive accuracy is misleading — a model that predicts "good loan" for everyone would score ~[X]%. The 62% recall on the minority class is the metric that maps to real-world value: catching defaults *before* they cost the lender.

## The Problem

Loan default datasets are heavily skewed — good loans dominate, bad loans are rare. A model optimized for accuracy will quietly learn to ignore the minority class. In a real lending context, that's the worst possible failure mode: every missed default costs the lender the entire principal, while every false alarm costs only the customer-acquisition friction of a manual review.

This project asks: *what does the pipeline look like when you optimize for the metric that actually matches the business cost?*

## Approach

End-to-end pipeline:

1. **Preprocessing** — handling missing values, dropping leakage-prone features, treating outliers
2. **Encoding** — categorical features encoded via [your method: target encoding / one-hot / label]
3. **Modeling** — LightGBM trained with class-weighted loss
4. **Threshold tuning** — sweeping the decision threshold and selecting the value that maximizes recall on the bad-loan class subject to a precision floor
5. **Evaluation** — stratified K-fold CV with PR-AUC, ROC-AUC, and per-class F1 — not just accuracy

## Data

- **Source:** LendingClub loan data, 2007–2014 (`loan_data_2007_2014.csv`)
- **Size:** [N] rows, [M] features
- **Target:** Binary version: `is_bad_loan` (default, charged-off, late). Multi-class version: loan status across [list categories].
- **Imbalance ratio:** ~[X]:1 (good:bad)

Datasets are too large for GitHub. Download links:
- [Main dataset](URL)
- [Train set](URL)
- [Test set](URL)

## The Key Decision: Recall Over Accuracy

The instinct on a 90%-accurate classifier is to ship it. I didn't, because the 10% it gets wrong is *almost entirely the class that matters*. Two options for fixing this:

1. **Resample** (SMOTE / undersampling) — risks injecting synthetic patterns the model treats as real
2. **Tune the decision threshold** — keeps the model honest, lets cost asymmetry drive the cutoff

I chose threshold tuning. The default 0.5 threshold gave [X]% recall on bad loans; sweeping the threshold and selecting for recall (with a precision floor of [Y]) pushed recall to **62%** while keeping overall accuracy at 89.5%. The F1 of 0.28 is the honest reflection of a still-hard minority-class problem — but it's the right tradeoff under the cost matrix that lending actually faces.

## Results in Detail

[Confusion matrix screenshot]
[PR curve screenshot]
[ROC curve screenshot]

## How to Run

```bash
git clone https://github.com/samuelfrann/Multi-Class-and-Binary-Loan-RIsk-Prediction.git
cd Multi-Class-and-Binary-Loan-RIsk-Prediction
pip install -r requirements.txt
# Download datasets to ./data/ (see Data section)
jupyter notebook notebooks/01_eda.ipynb
```

## Tech Stack

Python · LightGBM · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter

## What's Next

- Calibrated probabilities (Platt scaling / isotonic) for actual decision deployment
- Cost-sensitive learning with an explicit loss matrix
- SHAP-based explainability layer for adverse-action notices (regulatory relevance)
- Deploy as a FastAPI endpoint with monitoring on prediction drift

---

*Built by [Samuel Sotonwa](https://linkedin.com/in/samuel-sotonwa) — Mechatronics Engineer working on applied ML in insurance and risk.*
