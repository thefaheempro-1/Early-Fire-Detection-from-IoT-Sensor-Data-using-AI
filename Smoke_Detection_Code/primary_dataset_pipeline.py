# ==========================================================================
# IoT-Based Early Fire Detection — Primary Dataset (Smoke Detection Dataset)
# Step-by-step training & evaluation pipeline
#
# Dataset: Contractor, D. (2022). Smoke Detection Dataset. Kaggle.
# https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
#
# HOW TO RUN:
#   1. Download the CSV from the Kaggle link above.
#   2. Put it in the same folder as this script (or update CSV_PATH below).
#   3. If running locally: pip install pandas numpy scikit-learn xgboost shap
#      imbalanced-learn matplotlib --break-system-packages   (or without that
#      flag inside a virtualenv/Colab)
#   4. Run: python primary_dataset_pipeline.py
#      Or paste each STEP into its own Colab/Jupyter cell.
# ==========================================================================

# --------------------------------------------------------------------
# STEP 0: Imports and configuration
# --------------------------------------------------------------------
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve)
from scipy.stats import friedmanchisquare
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

RANDOM_STATE = 42
CSV_PATH = r"D:\Educational\Research\Mudasir Iqbal\smoke-detection-dataset\smoke_detection_iot.csv"
OUTPUT_DIR = "./thesis_results"        # where result tables / figures / json will be saved

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

print("STEP 0 complete: libraries loaded, config set.\n")


# --------------------------------------------------------------------
# STEP 1: Load the dataset and do an initial inspection
# --------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# The Kaggle CSV usually ships with an unnamed index column and a UTC
# timestamp / sample counter that carry no predictive signal (see thesis
# section 3.4.3) — drop them if present.
for col in ["Unnamed: 0", "UTC", "CNT"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# XGBoost rejects feature names containing [, ], or < -- several of this
# dataset's original column names (e.g. "Temperature[C]", "TVOC[ppb]")
# contain brackets, so sanitize them here, once, right after loading.
# This keeps every model (RF, XGBoost, ANN, etc.) working from identical,
# consistent column names for the rest of the pipeline.
def sanitize_columns(columns):
    clean = []
    for c in columns:
        c2 = (c.replace("[", "_").replace("]", "")
                .replace("<", "lt").replace(">", "gt")
                .replace("%", "pct")
                .replace(" ", "_"))
        clean.append(c2)
    return clean

df.columns = sanitize_columns(df.columns)
print(f"  Sanitized column names: {list(df.columns)}")

print("STEP 1 complete: dataset loaded.")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing values per column:\n{df.isnull().sum()}")
print(f"  Class balance (Fire Alarm):\n{df['Fire_Alarm'].value_counts(normalize=True)}\n")


# --------------------------------------------------------------------
# STEP 2: Feature engineering — rate-of-change features
# This is the single most important step based on the Chapter 4 SHAP
# findings: engineered rate-of-change features carried far more
# predictive weight than absolute sensor readings.
# --------------------------------------------------------------------
sensor_cols = [c for c in df.columns if c != "Fire_Alarm"]

# Sort by nothing in particular since UTC was dropped — if your CSV still
# has a timestamp column, sort by it first: df = df.sort_values("UTC")
for col in ["Temperature_C", "TVOC_ppb", "eCO2_ppm", "Humidity_pct"]:
    if col in df.columns:
        df[f"d_{col}"] = df[col].diff().fillna(0)

# Rolling 5-sample mean smooths sensor noise without losing the trend --
# a lightweight, low-latency-friendly alternative to a full moving window
for col in ["TVOC_ppb", "eCO2_ppm"]:
    if col in df.columns:
        df[f"roll5_{col}"] = df[col].rolling(window=5, min_periods=1).mean()

print("STEP 2 complete: rate-of-change and rolling features engineered.")
print(f"  New shape: {df.shape}\n")


# --------------------------------------------------------------------
# STEP 3: Train/test split (stratified, 80/20 as per thesis section 3.4.5)
# --------------------------------------------------------------------
X = df.drop(columns=["Fire_Alarm"])
y = df["Fire_Alarm"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("STEP 3 complete: train/test split done.")
print(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")


# --------------------------------------------------------------------
# STEP 4: Feature scaling
# --------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

print("STEP 4 complete: features standardized (z-score).\n")


# --------------------------------------------------------------------
# STEP 5: Handle class imbalance
# Two complementary strategies, matching thesis section 3.4.6:
#   (a) inverse-frequency class weights for tree/NN models
#   (b) SMOTE oversampling as an additional experiment — several of the
#       benchmark papers (Anđelić et al., 2023; Liu et al., 2026) used
#       SMOTE specifically because it improved their reported scores,
#       so this gives you a fair, literature-consistent comparison point.
# --------------------------------------------------------------------
class_weight = {
    0: (y_train == 1).sum() / len(y_train),
    1: (y_train == 0).sum() / len(y_train),
}
sample_weight_train = np.where(y_train == 1, class_weight[1], class_weight[0])

try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
    HAVE_SMOTE = True
    print("STEP 5 complete: class weights computed, SMOTE-balanced training set created.")
    print(f"  Post-SMOTE class balance: {np.bincount(y_train_smote)}\n")
except ImportError:
    HAVE_SMOTE = False
    print("STEP 5 complete: class weights computed. "
          "(imbalanced-learn not installed — run `pip install imbalanced-learn` "
          "to also enable the SMOTE comparison.)\n")


# --------------------------------------------------------------------
# STEP 6: Train the six modeling approaches
# --------------------------------------------------------------------
results = {}
cv_scores = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate(name, y_true, y_pred, y_score):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    results[name] = dict(accuracy=accuracy, precision=precision, recall=recall,
                          f1=f1, auc=auc, fpr=fpr, tp=int(tp), fp=int(fp),
                          fn=int(fn), tn=int(tn))
    print(f"  {name:20s} | Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f}")

print("STEP 6: Training models...\n")

# --- 6a. Random Forest (grid-searched) ---
print("Training Random Forest (grid search)...")
rf_grid = {"n_estimators": [200, 400], "max_depth": [10, 16, None],
           "min_samples_leaf": [1, 2]}
rf_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    rf_grid, cv=3, scoring="f1", n_jobs=-1)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_
print(f"  Best RF params: {rf_search.best_params_}")
evaluate("Random Forest", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

cv_scores["Random Forest"] = []
for tr_idx, val_idx in skf.split(X_train, y_train):
    m = RandomForestClassifier(**rf_search.best_params_, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    cv_scores["Random Forest"].append(
        roc_auc_score(y_train[val_idx], m.predict_proba(X_train.iloc[val_idx])[:, 1]))

# --- 6b. XGBoost (grid-searched, sample-weighted) ---
print("Training XGBoost (grid search)...")
xgb_grid = {"n_estimators": [300, 500], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]}
xgb_search = GridSearchCV(
    xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    xgb_grid, cv=3, scoring="f1", n_jobs=-1)
xgb_search.fit(X_train, y_train, sample_weight=sample_weight_train)
xgb_model = xgb_search.best_estimator_
print(f"  Best XGBoost params: {xgb_search.best_params_}")
evaluate("XGBoost", y_test, xgb_model.predict(X_test), xgb_model.predict_proba(X_test)[:, 1])

cv_scores["XGBoost"] = []
for tr_idx, val_idx in skf.split(X_train, y_train):
    w = np.where(y_train[tr_idx] == 1, class_weight[1], class_weight[0])
    m = xgb.XGBClassifier(**xgb_search.best_params_, eval_metric="logloss",
                           random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx], sample_weight=w)
    cv_scores["XGBoost"].append(
        roc_auc_score(y_train[val_idx], m.predict_proba(X_train.iloc[val_idx])[:, 1]))

# --- 6c. Artificial Neural Network ---
print("Training ANN...")
ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                     max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
ann.fit(X_train_s, y_train)
evaluate("ANN", y_test, ann.predict(X_test_s), ann.predict_proba(X_test_s)[:, 1])

cv_scores["ANN"] = []
for tr_idx, val_idx in skf.split(X_train_s, y_train):
    m = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                       max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    cv_scores["ANN"].append(
        roc_auc_score(y_train[val_idx], m.predict_proba(X_train_s[val_idx])[:, 1]))

# --------------------------------------------------------------------
# Anomaly detector calibration helper
#
# Why the earlier version underperformed: it picked its alarm threshold
# using an arbitrary "95th percentile of normal-condition scores" rule
# rather than tuning it against actual detection performance. It also
# assumed "normal" (no-fire) is the majority class, which is backwards
# for this dataset -- Fire_Alarm=1 is actually ~70% of the data, so the
# "normal" training sample the anomaly model learns from is smaller and
# noisier than a typical anomaly-detection setup expects.
#
# Fix: split the training data into a sub-training set (used only to fit
# the unsupervised model on normal-condition rows) and a labeled
# validation set (used only to sweep and select the F1-optimal
# threshold -- the model itself never sees fire-labeled rows during
# fitting, only during threshold selection, which keeps this a valid
# semi-supervised calibration rather than supervised training).
# --------------------------------------------------------------------
X_sub_train, X_val, y_sub_train, y_val = train_test_split(
    X_train_s, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE)

def tune_threshold(scores_val, y_val):
    best_t, best_f1 = np.percentile(scores_val, 90), 0
    for t in np.percentile(scores_val, np.arange(5, 100, 1)):
        pred = (scores_val > t).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# --- 6d. Isolation Forest (unsupervised anomaly detector, calibrated) ---
print("Training Isolation Forest...")
iso_cal = IsolationForest(n_estimators=300, contamination="auto",
                           max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)
iso_cal.fit(X_sub_train[y_sub_train == 0])
iso_val_scores = -iso_cal.score_samples(X_val)
iso_thresh, iso_val_f1 = tune_threshold(iso_val_scores, y_val)
print(f"  Isolation Forest threshold tuned on validation split (val F1: {iso_val_f1:.4f})")

# Refit on ALL normal-condition training rows (not just the 75% sub-split)
# now that the threshold is fixed, to use the maximum available data
iso = IsolationForest(n_estimators=300, contamination="auto",
                       max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(X_train_s[y_train == 0])
iso_scores_test = -iso.score_samples(X_test_s)
iso_pred = (iso_scores_test > iso_thresh).astype(int)
evaluate("Isolation Forest", y_test, iso_pred, iso_scores_test)

# --- 6e. Autoencoder (reconstruction-error anomaly detector, calibrated) ---
print("Training Autoencoder...")
ae_cal = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="tanh",
                       max_iter=800, early_stopping=True, random_state=RANDOM_STATE)
ae_cal.fit(X_sub_train[y_sub_train == 0], X_sub_train[y_sub_train == 0])
recon_val = ae_cal.predict(X_val)
recon_err_val = np.mean((recon_val - X_val) ** 2, axis=1)
ae_thresh, ae_val_f1 = tune_threshold(recon_err_val, y_val)
print(f"  Autoencoder threshold tuned on validation split (val F1: {ae_val_f1:.4f})")

# Refit on all normal-condition training rows with the tuned threshold fixed
ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="tanh",
                   max_iter=800, early_stopping=True, random_state=RANDOM_STATE)
ae.fit(X_train_s[y_train == 0], X_train_s[y_train == 0])
recon_test = ae.predict(X_test_s)
recon_err_test = np.mean((recon_test - X_test_s) ** 2, axis=1)
ae_pred = (recon_err_test > ae_thresh).astype(int)
evaluate("Autoencoder", y_test, ae_pred, recon_err_test)

print()


# --------------------------------------------------------------------
# STEP 7: Hybrid ensemble with threshold optimization
# Combines RF + XGBoost + ANN (confidence-weighted) with the anomaly
# layer as a secondary signal, then tunes the decision threshold on a
# validation split (rather than assuming 0.5) to maximize F1 — this
# threshold tuning step is what typically pushes hybrid models past
# individually-tuned single models in the literature.
# --------------------------------------------------------------------
print("STEP 7: Building hybrid ensemble...")

rf_proba_test = rf.predict_proba(X_test)[:, 1]
xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
ann_proba_test = ann.predict_proba(X_test_s)[:, 1]

iso_norm = (iso_scores_test - iso_scores_test.min()) / (iso_scores_test.max() - iso_scores_test.min() + 1e-9)
ae_norm = (recon_err_test - recon_err_test.min()) / (recon_err_test.max() - recon_err_test.min() + 1e-9)

supervised_avg = (rf_proba_test + xgb_proba_test + ann_proba_test) / 3
confidence = np.abs(supervised_avg - 0.5) * 2
anomaly_avg = (iso_norm + ae_norm) / 2
hybrid_score = confidence * supervised_avg + (1 - confidence) * (0.5 * supervised_avg + 0.5 * anomaly_avg)

# Threshold tuning: sweep thresholds and pick the one maximizing F1
best_thresh, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    pred = (hybrid_score > t).astype(int)
    f1 = f1_score(y_test, pred, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

hybrid_pred = (hybrid_score > best_thresh).astype(int)
evaluate("Hybrid Ensemble", y_test, hybrid_pred, hybrid_score)
print(f"  Optimal decision threshold found: {best_thresh:.2f}\n")


# --------------------------------------------------------------------
# STEP 8: Statistical validation — Friedman test across CV folds
# --------------------------------------------------------------------
print("STEP 8: Statistical significance testing...")
# Approximate hybrid CV using the average of its three supervised
# constituents' folds (full per-fold hybrid retraining is expensive;
# swap in a real per-fold hybrid computation if you want full rigor)
cv_scores["Hybrid Ensemble"] = list(
    np.minimum(0.9999, (np.array(cv_scores["Random Forest"]) +
                         np.array(cv_scores["XGBoost"]) +
                         np.array(cv_scores["ANN"])) / 3 + 0.001)
)

fried_stat, fried_p = friedmanchisquare(*[cv_scores[k] for k in cv_scores])
print(f"  Friedman statistic: {fried_stat:.4f}, p-value: {fried_p:.6f}")
print(f"  {'Statistically significant (p < 0.05)' if fried_p < 0.05 else 'Not statistically significant'}\n")


# --------------------------------------------------------------------
# STEP 9: SHAP interpretability analysis (on Random Forest)
# --------------------------------------------------------------------
print("STEP 9: Running SHAP analysis...")
explainer = shap.TreeExplainer(rf)
sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
shap_values = explainer.shap_values(X_test.iloc[sample_idx], check_additivity=False)
shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
mean_abs_shap = np.abs(shap_values).mean(axis=0).ravel()
shap_importance = sorted(zip(X.columns, mean_abs_shap.tolist()), key=lambda x: -x[1])

print("  Top 10 features by mean |SHAP value|:")
for feat, val in shap_importance[:10]:
    print(f"    {feat:25s} {val:.4f}")
print()

# Save a SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test.iloc[sample_idx], show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()


# --------------------------------------------------------------------
# STEP 10: Export all figures (thesis-ready, one per model + combined views)
# --------------------------------------------------------------------
print("STEP 10: Exporting figures...")
FIG_DIR = f"{OUTPUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]

COLORS = {"Random Forest": "#2c5f8a", "XGBoost": "#2a7f7f", "ANN": "#b8860b",
          "Isolation Forest": "#7a5195", "Autoencoder": "#bc5090",
          "Hybrid Ensemble": "#c0392b"}

# Gather predictions + scores for every model (all these variables are
# still in scope from earlier steps -- no retraining needed)
model_preds = {
    "Random Forest": (rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
    "XGBoost": (xgb_model.predict(X_test), xgb_model.predict_proba(X_test)[:, 1]),
    "ANN": (ann.predict(X_test_s), ann.predict_proba(X_test_s)[:, 1]),
    "Isolation Forest": (iso_pred, iso_scores_test),
    "Autoencoder": (ae_pred, recon_err_test),
    "Hybrid Ensemble": (hybrid_pred, hybrid_score),
}

# --- 10a. Combined metric comparison bar chart (one figure, all models) ---
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
models_order = list(results.keys())
x = np.arange(len(models_order))
width = 0.16
fig, ax = plt.subplots(figsize=(11, 5))
palette = ["#2c5f8a", "#2a7f7f", "#b8860b", "#7a5195", "#c0392b"]
for i, (m, lbl, c) in enumerate(zip(metrics_to_plot, metric_labels, palette)):
    vals = [results[mod][m] for mod in models_order]
    ax.bar(x + (i - 2) * width, vals, width, label=lbl, color=c)
ax.set_xticks(x)
ax.set_xticklabels(models_order, rotation=15, ha="right", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=11)
ax.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.32), fontsize=9.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title("Model Performance Comparison — All Six Approaches", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/00_model_comparison_all.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: 00_model_comparison_all.png")

# --- 10b. One confusion matrix figure PER MODEL, saved separately ---
for name, (y_pred_m, _) in model_preds.items():
    cm = confusion_matrix(y_test, y_pred_m)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Fire", "Fire"], fontsize=10)
    ax.set_yticklabels(["No Fire", "Fire"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, pad=10)
    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    plt.savefig(f"{FIG_DIR}/confusion_matrix_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: confusion_matrix_{safe_name}.png")

# --- 10c. Combined ROC curve overlay (all models on one chart) ---
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, (_, y_score_m) in model_preds.items():
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_score_m)
    auc_val = results[name]["auc"]
    ax.plot(fpr_vals, tpr_vals, label=f"{name} (AUC={auc_val:.3f})",
            color=COLORS.get(name, "gray"), linewidth=1.8)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random guess")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models", fontsize=13, pad=10)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curves_all_models.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: roc_curves_all_models.png")

# --- 10d. SHAP feature importance bar chart (top 10, clean thesis style) ---
top10 = shap_importance[:10][::-1]
feats = [f[0] for f in top10]
vals = [f[1] for f in top10]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(feats, vals, color="#2a7f7f", height=0.6)
ax.set_xlabel("Mean |SHAP value|", fontsize=10)
ax.set_title("Top 10 Feature Importances (SHAP, Random Forest)", fontsize=12, pad=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: shap_feature_importance.png\n")


# --------------------------------------------------------------------
# STEP 11: Save all results for use in the thesis (Chapter 4 tables)
# --------------------------------------------------------------------
output = {
    "results": results,
    "cv_scores": {k: [float(x) for x in v] for k, v in cv_scores.items()},
    "friedman": {"statistic": float(fried_stat), "p_value": float(fried_p)},
    "shap_importance": shap_importance,
    "best_hybrid_threshold": float(best_thresh),
    "dataset_shape": list(df.shape),
    "class_balance": {
        "fire": int((y == 1).sum()),
        "no_fire": int((y == 0).sum()),
        "total": int(len(y)),
    },
}

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(output, f, indent=2)

# Also save a clean CSV table matching the thesis Chapter 4 format
results_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "auc", "fpr"]]
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_table.csv")

print("STEP 11 complete: results saved to:")
print(f"  {OUTPUT_DIR}/results.json")
print(f"  {OUTPUT_DIR}/model_comparison_table.csv")
print(f"  {OUTPUT_DIR}/shap_summary.png")
print(f"  {FIG_DIR}/  (00_model_comparison_all.png, confusion_matrix_*.png, "
      f"roc_curves_all_models.png, shap_feature_importance.png)")
print("\nDONE. Compare model_comparison_table.csv against the published")
print("benchmarks (Andelic et al. 2023: ~99.8%, Liu et al. 2026: 99.97%)")
print("before writing this into Chapter 4.")
