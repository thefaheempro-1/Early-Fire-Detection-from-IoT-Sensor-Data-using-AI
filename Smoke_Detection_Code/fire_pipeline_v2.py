# ==========================================================================
# Early Fire Detection from Multi-Sensor IoT Data -- CORRECTED PIPELINE (v2)
# Dataset: Smoke Detection Dataset (Contractor, 2022), smoke_detection_iot.csv
#
# What this fixes compared with the earlier scripts
#   1. LEAKAGE: features (diff / rolling) are computed INSIDE contiguous time
#      blocks, and whole blocks go to train / val / test. Neighbouring rows can
#      no longer sit on both sides of a split.
#   2. TEST-SET TUNING: every threshold (anomaly detectors, hybrid) and every
#      normalisation constant is chosen on a separate validation split. The
#      test split is only used for final scoring.
#   3. REAL CROSS-VALIDATION for ALL models (incl. Isolation Forest, Autoencoder
#      and the hybrid) with block-wise folds -> honest Friedman test.
#   4. BASELINES: majority class, logistic regression, depth-3 tree, best single
#      sensor threshold.
#   5. ABLATIONS: with/without Pressure, with/without engineered features.
#   6. Same trained detector scored under BOTH threshold rules
#      (95th-percentile-of-normal rule vs validation-tuned rule).
#   7. DETECTION LATENCY study (time from fire onset to first alarm) and
#      inference time per sample.
#   8. Legacy random-row split shown next to the block split so the optimism
#      gap caused by leakage is measured, not guessed.
#   9. Multiple split seeds -> mean +/- std instead of one lucky split.
#
# RUN:   python fire_pipeline_v2.py
#        python fire_pipeline_v2.py --quick        (fast smoke run, ~5 min)
#        python fire_pipeline_v2.py --csv "path\to\smoke_detection_iot.csv"
# NEEDS: pip install pandas numpy scikit-learn xgboost scipy matplotlib shap
# OUTPUT: ./results_v2/  (CSV tables, results.json, figures/, run_log.txt)
# ==========================================================================
import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             matthews_corrcoef, brier_score_loss)
from sklearn.model_selection import train_test_split
from scipy.stats import friedmanchisquare, rankdata
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- configuration -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=r"D:\Educational\Research\02 Fire Detection Paper\smoke-detection-dataset\smoke_detection_iot.csv")
ap.add_argument("--out", default="./results_v2")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--block", type=int, default=600, help="rows per time block (~10 min at 1 Hz)")
args = ap.parse_args()

CSV_PATH = args.csv
OUT = args.out
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
QUICK = args.quick
BLOCK = args.block
SEEDS = [42, 43] if QUICK else [42, 43, 44, 45, 46]
CV_FOLDS = 5
CV_REPEATS = 1 if QUICK else 2
N_TREES = 100 if QUICK else 300
FRACS = (0.6, 0.2, 0.2)  # train / val / test (by rows, whole blocks)

LOG = open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()

# ----------------------------- data loading ------------------------------
def sanitize(c):
    return (c.replace("[", "_").replace("]", "").replace("%", "pct")
             .replace("<", "lt").replace(">", "gt").replace(" ", "_").replace(".", "_"))

def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df.columns = [sanitize(c) for c in df.columns]
    assert "Fire_Alarm" in df.columns, f"Fire_Alarm column not found: {list(df.columns)}"
    df["UTC"] = pd.to_numeric(df["UTC"], errors="coerce")
    sort_cols = ["UTC"] + (["CNT"] if "CNT" in df.columns else [])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["y"] = df["Fire_Alarm"].astype(int)
    # segments = recording sessions (gap > 60 s), blocks = BLOCK-row chunks inside a segment
    seg = (df["UTC"].diff().fillna(0) > 60).cumsum()
    pos = df.groupby(seg).cumcount()
    df["block"] = seg.astype(str) + "_" + (pos // BLOCK).astype(str)
    log(f"Loaded {len(df)} rows, {df['block'].nunique()} blocks, {seg.nunique()} recording segments")
    log(f"Fire prevalence: {df['y'].mean():.4f}; exact duplicate sensor rows: "
        f"{int(df.drop(columns=['block','y','UTC'] + (['CNT'] if 'CNT' in df.columns else [])).duplicated().sum())}")
    # Causal engineered features, computed WITHIN each block only (no cross-split leakage)
    g = df.groupby("block")
    for col in ["Temperature_C", "TVOC_ppb", "eCO2_ppm", "Humidity_pct"]:
        if col in df.columns:
            df["d_" + col] = g[col].diff().fillna(0)
    for col in ["TVOC_ppb", "eCO2_ppm"]:
        if col in df.columns:
            df["roll5_" + col] = g[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return df

df = load_data()
NON_FEATURES = {"Fire_Alarm", "y", "UTC", "CNT", "block"}  # CNT/UTC excluded: counters/timestamps leak session position
ALL_FEATS = [c for c in df.columns if c not in NON_FEATURES]
ENG_FEATS = [f for f in ALL_FEATS if f.startswith(("d_", "roll5_"))]
RAW_FEATS = [f for f in ALL_FEATS if f not in ENG_FEATS]
log("Features:", ALL_FEATS)

# ----------------------------- splitting ---------------------------------
BLOCK_SIZES = df["block"].value_counts().to_dict()
ALL_BLOCKS = list(df["block"].unique())  # time order of first appearance

def partition_blocks(blocks, fracs, rng=None):
    blocks = list(blocks)
    if rng is not None:
        blocks = list(np.array(blocks)[rng.permutation(len(blocks))])
    total = sum(BLOCK_SIZES[b] for b in blocks)
    cum = np.cumsum(fracs) * total
    parts = [[] for _ in fracs]; run = 0; k = 0
    for b in blocks:
        while k < len(fracs) - 1 and run >= cum[k]:
            k += 1
        parts[k].append(b); run += BLOCK_SIZES[b]
    return parts

def frame(blocks):
    return df[df["block"].isin(set(blocks))]  # preserves chronological order

def block_split(seed, fracs=FRACS):
    rng = np.random.default_rng(seed)
    tr, va, te = partition_blocks(ALL_BLOCKS, fracs, rng)
    return frame(tr), frame(va), frame(te)

def chrono_split(fracs=FRACS):
    tr, va, te = partition_blocks(ALL_BLOCKS, fracs, None)
    return frame(tr), frame(va), frame(te)

def ok_split(*parts):
    return all(p["y"].nunique() == 2 and len(p) > 50 for p in parts)

# ----------------------------- metrics -----------------------------------
def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def metrics(y, pred, score, prob=False):
    y = np.asarray(y); pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(accuracy=(tp + tn) / len(y),
               precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0),
               auc=roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
               fpr=fp / (fp + tn) if (fp + tn) else np.nan,
               mcc=matthews_corrcoef(y, pred),
               tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    if prob:
        s = np.clip(score, 0, 1)
        out["brier"] = brier_score_loss(y, s); out["ece"] = ece_score(y, s)
    return {k: float(v) for k, v in out.items()}

def best_f1_threshold(score, y):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])

# ----------------------------- model suite -------------------------------
PROB_MODELS = {"Random Forest", "XGBoost", "ANN", "Logistic Regression", "Supervised Ensemble (no anomaly)", "Hybrid Ensemble"}

class Suite:
    """Fits every model on `tr`; tunes ALL thresholds / normalisations on `va` only."""
    def __init__(self, feats, seed):
        self.f, self.seed = feats, seed

    def fit(self, tr, va):
        f, s = self.f, self.seed
        Xtr, ytr, Xva, yva = tr[f].values, tr["y"].values, va[f].values, va["y"].values
        self.sc = StandardScaler().fit(Xtr)
        Xs, Vs = self.sc.transform(Xtr), self.sc.transform(Xva)
        pos = ytr.mean(); spw = (1 - pos) / max(pos, 1e-9)
        self.majority = int(pos >= 0.5)
        self.rf = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2, class_weight="balanced",
                                         random_state=s, n_jobs=-1).fit(Xtr, ytr)
        self.xgb = xgb.XGBClassifier(n_estimators=N_TREES, max_depth=6, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                     tree_method="hist", random_state=s, n_jobs=-1).fit(Xtr, ytr)
        self.ann = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300,
                                 random_state=s).fit(Xs, ytr)
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, ytr)
        self.dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=s).fit(Xtr, ytr)
        # semi-supervised anomaly detectors: fitted ONLY on label-0 (no-fire) training rows
        normal = Xs[ytr == 0]
        self.iso = IsolationForest(n_estimators=N_TREES, max_features=0.7, random_state=s, n_jobs=-1).fit(normal)
        self.ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="tanh", early_stopping=True,
                               max_iter=500, random_state=s).fit(normal, normal)
        # single-feature baseline: pick best feature by train AUC, threshold on val
        best = (-1, None)
        for j, name in enumerate(f):
            a = roc_auc_score(ytr, Xtr[:, j]); sign = 1 if a >= 0.5 else -1
            t, f1v = best_f1_threshold(sign * Xva[:, j], yva)
            if f1v > best[0]:
                best = (f1v, (j, name, sign, t))
        self.sf_j, self.sf_name, self.sf_sign, self.sf_t = best[1]
        # anomaly thresholds: tuned rule (val) and naive rule (95th pct of normal TRAIN scores)
        iso_tr, ae_tr = self._iso(Xs), self._ae(Xs)
        self.iso_lo, self.iso_hi = np.percentile(iso_tr, [1, 99])
        self.ae_lo, self.ae_hi = np.percentile(ae_tr, [1, 99])
        self.iso_t, _ = best_f1_threshold(self._iso(Vs), yva)
        self.ae_t, _ = best_f1_threshold(self._ae(Vs), yva)
        self.iso_naive = float(np.percentile(iso_tr[ytr == 0], 95))
        self.ae_naive = float(np.percentile(ae_tr[ytr == 0], 95))
        # hybrid threshold: tuned on val (out-of-sample for all components)
        sv = self.scores(va)
        self.hyb_t, self.hyb_val_f1 = best_f1_threshold(sv["Hybrid Ensemble"], yva)
        self.sup_t, _ = best_f1_threshold(sv["Supervised Ensemble (no anomaly)"], yva)
        return self

    def _iso(self, Xs): return -self.iso.score_samples(Xs)
    def _ae(self, Xs): return np.mean((self.ae.predict(Xs) - Xs) ** 2, axis=1)

    def scores(self, d):
        X = d[self.f].values; Xs = self.sc.transform(X)
        rf, xg, an = (self.rf.predict_proba(X)[:, 1], self.xgb.predict_proba(X)[:, 1], self.ann.predict_proba(Xs)[:, 1])
        iso, ae = self._iso(Xs), self._ae(Xs)
        isoN = np.clip((iso - self.iso_lo) / (self.iso_hi - self.iso_lo + 1e-9), 0, 1)
        aeN = np.clip((ae - self.ae_lo) / (self.ae_hi - self.ae_lo + 1e-9), 0, 1)
        sup = (rf + xg + an) / 3; conf = np.abs(sup - 0.5) * 2; anom = (isoN + aeN) / 2
        hyb = conf * sup + (1 - conf) * (0.5 * sup + 0.5 * anom)
        return {"Random Forest": rf, "XGBoost": xg, "ANN": an,
                "Logistic Regression": self.lr.predict_proba(Xs)[:, 1],
                "Decision Tree (d=3)": self.dt.predict_proba(X)[:, 1],
                "Isolation Forest": iso, "Autoencoder": ae,
                "Supervised Ensemble (no anomaly)": sup, "Hybrid Ensemble": hyb,
                "Isolation Forest (95th-pct rule)": iso, "Autoencoder (95th-pct rule)": ae,
                f"Best Single Feature": self.sf_sign * X[:, self.sf_j],
                "Majority Class": np.full(len(X), 0.5)}

    def predict(self, d, sc=None):
        sc = sc or self.scores(d)
        thr = {"Random Forest": .5, "XGBoost": .5, "ANN": .5, "Logistic Regression": .5, "Decision Tree (d=3)": .5,
               "Isolation Forest": self.iso_t, "Autoencoder": self.ae_t,
               "Supervised Ensemble (no anomaly)": self.sup_t, "Hybrid Ensemble": self.hyb_t,
               "Isolation Forest (95th-pct rule)": self.iso_naive, "Autoencoder (95th-pct rule)": self.ae_naive,
               "Best Single Feature": self.sf_t}
        pred = {k: (sc[k] >= t).astype(int) for k, t in thr.items()}
        pred["Majority Class"] = np.full(len(d), self.majority)
        return sc, pred

    def evaluate(self, d):
        sc, pred = self.predict(d)
        return {k: metrics(d["y"].values, pred[k], sc[k], prob=k in PROB_MODELS) for k in sc}, sc, pred

# ----------------------------- helpers -----------------------------------
def agg_table(runs, metric_list=("accuracy", "precision", "recall", "f1", "auc", "fpr", "mcc")):
    models = list(runs[0].keys()); rows = []
    for m in models:
        row = {"model": m}
        for k in metric_list:
            v = np.array([r[m][k] for r in runs], dtype=float)
            row[k + "_mean"], row[k + "_std"] = np.nanmean(v), np.nanstd(v)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

def pretty(t, cols=("accuracy", "precision", "recall", "f1", "auc", "fpr")):
    o = pd.DataFrame(index=t.index)
    for c in cols:
        o[c] = [f"{a:.4f} +/- {b:.4f}" for a, b in zip(t[c + "_mean"], t[c + "_std"])]
    return o

def detection_latency(te, pred):
    y, t, b = te["y"].values, te["UTC"].values.astype(float), te["block"].values
    lat, missed, n, i = [], 0, len(y), 1
    while i < n:
        if y[i] == 1 and y[i - 1] == 0 and b[i] == b[i - 1]:
            j = i
            while j < n and y[j] == 1 and b[j] == b[i]:
                j += 1
            hit = np.where(pred[i:j] == 1)[0]
            if len(hit): lat.append(t[i + hit[0]] - t[i])
            else: missed += 1
            i = j
        else:
            i += 1
    return lat, missed

# ============================== EXPERIMENTS ===============================
t0 = time.time()
results = {}

# ---- E1: main experiment, block split, several seeds ----
log("\n=== E1: block-wise train/val/test split, seeds", SEEDS, "===")
main_runs, lat_pool, saved = [], {}, {}
for s in SEEDS:
    tr, va, te = block_split(s)
    if not ok_split(tr, va, te):
        log(f"  seed {s}: a split lacks one class, skipped"); continue
    suite = Suite(ALL_FEATS, s).fit(tr, va)
    res, sc, pred = suite.evaluate(te)
    main_runs.append(res)
    for k in pred:
        l, miss = detection_latency(te, pred[k])
        d = lat_pool.setdefault(k, {"lat": [], "missed": 0}); d["lat"] += l; d["missed"] += miss
    if not saved:
        saved = dict(suite=suite, te=te, sc=sc, pred=pred)
    log(f"  seed {s}: train/val/test = {len(tr)}/{len(va)}/{len(te)} | "
        f"hybrid F1={res['Hybrid Ensemble']['f1']:.4f} AUC={res['Hybrid Ensemble']['auc']:.4f} "
        f"RF F1={res['Random Forest']['f1']:.4f} | hybrid thr={suite.hyb_t:.3f} (tuned on val)")
main_tab = agg_table(main_runs)
main_tab.to_csv(os.path.join(OUT, "E1_main_block_split.csv"))
pretty(main_tab).to_csv(os.path.join(OUT, "E1_main_block_split_pretty.csv"))
log(pretty(main_tab).to_string())
results["E1_main"] = main_tab.to_dict(orient="index")
extra = ["brier", "ece"]
cal_tab = agg_table([{k: v for k, v in r.items() if k in PROB_MODELS} for r in main_runs], tuple(extra))
cal_tab.to_csv(os.path.join(OUT, "E1_probability_calibration.csv"))
log("\nProbability calibration (Brier / ECE):\n" + cal_tab.round(4).to_string())

# ---- E2: threshold-rule comparison on the SAME trained detector ----
log("\n=== E2: same detector, two threshold rules (95th-pct rule vs validation-tuned) ===")
rows = []
for base in ["Isolation Forest", "Autoencoder"]:
    for label, key in [("95th-pct rule", base + " (95th-pct rule)"), ("val-tuned", base)]:
        v = {k: np.mean([r[key][k] for r in main_runs]) for k in ("precision", "recall", "f1", "auc", "fpr")}
        rows.append(dict(detector=base, rule=label, **v))
thr_tab = pd.DataFrame(rows); thr_tab.to_csv(os.path.join(OUT, "E2_threshold_rule_comparison.csv"), index=False)
log(thr_tab.round(4).to_string(index=False))
log("(AUC is identical across rules by construction: a threshold cannot change AUC.)")

# ---- E3: chronological hold-out (harder, deployment-like) ----
log("\n=== E3: chronological split (train on earliest blocks, test on latest) ===")
tr, va, te = chrono_split()
if ok_split(tr, va, te):
    suite = Suite(ALL_FEATS, 42).fit(tr, va)
    res, _, _ = suite.evaluate(te)
    chrono_tab = pd.DataFrame(res).T[["accuracy", "precision", "recall", "f1", "auc", "fpr", "mcc"]]
    chrono_tab.to_csv(os.path.join(OUT, "E3_chronological_split.csv"))
    log(chrono_tab.round(4).to_string())
    results["E3_chrono"] = chrono_tab.to_dict(orient="index")
else:
    log("  chronological split lacks a class in some part; skipped (report this as a dataset limitation)")

# ---- E4: real block-wise cross-validation for ALL models + Friedman ----
log(f"\n=== E4: {CV_FOLDS}-fold x {CV_REPEATS} block-wise CV, all models ===")
cv_f1, cv_auc = {}, {}
for rep in range(CV_REPEATS):
    rng = np.random.default_rng(1000 + rep)
    folds = partition_blocks(ALL_BLOCKS, (1.0 / CV_FOLDS,) * CV_FOLDS, rng)
    for k, fold_blocks in enumerate(folds):
        te = frame(fold_blocks)
        rest = [b for b in ALL_BLOCKS if b not in set(fold_blocks)]
        trb, vab = partition_blocks(rest, (0.8, 0.2), np.random.default_rng(2000 + rep * 10 + k))
        tr, va = frame(trb), frame(vab)
        if not ok_split(tr, va, te):
            log(f"  rep {rep} fold {k}: a part lacks one class, skipped"); continue
        res, _, _ = Suite(ALL_FEATS, 42 + rep).fit(tr, va).evaluate(te)
        for m, r in res.items():
            cv_f1.setdefault(m, []).append(r["f1"]); cv_auc.setdefault(m, []).append(r["auc"])
        log(f"  rep {rep} fold {k}: hybrid F1={res['Hybrid Ensemble']['f1']:.4f}")
cv_models = [m for m in cv_f1 if m != "Majority Class" and "95th" not in m]
def friedman(d, name):
    M = np.array([d[m] for m in cv_models])  # models x folds
    try:
        stat, p = friedmanchisquare(*M)
    except Exception:
        stat, p = np.nan, np.nan
    ranks = np.mean([rankdata(-M[:, j]) for j in range(M.shape[1])], axis=0)
    log(f"  Friedman on {name}: chi2={stat:.3f}, p={p:.5f}, folds={M.shape[1]}, models={len(cv_models)}")
    return dict(statistic=float(stat), p=float(p), avg_rank=dict(zip(cv_models, map(float, ranks))))
fr = {"f1": friedman(cv_f1, "F1"), "auc": friedman(cv_auc, "AUC")}
cv_tab = pd.DataFrame({m: {"f1_mean": np.mean(cv_f1[m]), "f1_std": np.std(cv_f1[m]),
                           "auc_mean": np.nanmean(cv_auc[m]), "auc_std": np.nanstd(cv_auc[m])} for m in cv_f1}).T
cv_tab.to_csv(os.path.join(OUT, "E4_cv_summary.csv")); log(cv_tab.round(4).to_string())
pd.DataFrame(cv_f1).to_csv(os.path.join(OUT, "E4_cv_f1_per_fold.csv"), index=False)
pd.DataFrame(cv_auc).to_csv(os.path.join(OUT, "E4_cv_auc_per_fold.csv"), index=False)
results["E4_cv"] = dict(summary=cv_tab.to_dict(orient="index"), friedman=fr)

# ---- E5: ablations (RF + XGBoost, block splits) ----
log("\n=== E5: feature ablations ===")
sets = {"All features": ALL_FEATS,
        "No Pressure": [f for f in ALL_FEATS if f != "Pressure_hPa"],
        "Raw only (no engineered)": RAW_FEATS,
        "Raw only, no Pressure": [f for f in RAW_FEATS if f != "Pressure_hPa"],
        "Engineered only": ENG_FEATS,
        "Pressure only": ["Pressure_hPa"]}
abl_rows = []
for name, feats in sets.items():
    if not feats: continue
    for mdl in ["Random Forest", "XGBoost"]:
        f1s, aucs = [], []
        for s in SEEDS:
            tr, va, te = block_split(s)
            if not ok_split(tr, va, te): continue
            Xtr, ytr, Xte, yte = tr[feats].values, tr["y"].values, te[feats].values, te["y"].values
            if mdl == "Random Forest":
                m = RandomForestClassifier(n_estimators=N_TREES // 2, min_samples_leaf=2, class_weight="balanced",
                                           random_state=s, n_jobs=-1).fit(Xtr, ytr)
            else:
                m = xgb.XGBClassifier(n_estimators=N_TREES // 2, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                      tree_method="hist", random_state=s, n_jobs=-1).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            f1s.append(f1_score(yte, p >= .5, zero_division=0)); aucs.append(roc_auc_score(yte, p))
        abl_rows.append(dict(feature_set=name, n_features=len(feats), model=mdl,
                             f1=f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", auc=f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"))
abl = pd.DataFrame(abl_rows); abl.to_csv(os.path.join(OUT, "E5_ablation.csv"), index=False)
log(abl.to_string(index=False))

# ---- E6: legacy random-row split vs block split (quantifies leakage optimism) ----
log("\n=== E6: legacy random-row split (leaky) vs block split ===")
leg = []
for s in SEEDS:
    tr_i, te_i = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["y"], random_state=s)
    Xtr, ytr, Xte, yte = df[ALL_FEATS].values[tr_i], df["y"].values[tr_i], df[ALL_FEATS].values[te_i], df["y"].values[te_i]
    m = RandomForestClassifier(n_estimators=N_TREES // 2, min_samples_leaf=2, class_weight="balanced",
                               random_state=s, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    leg.append((f1_score(yte, p >= .5), roc_auc_score(yte, p)))
leg = np.array(leg)
rf_blk = [r["Random Forest"] for r in main_runs]
e6 = pd.DataFrame({"split": ["random-row (leaky)", "block-wise (honest)"],
                   "RF_F1": [f"{leg[:,0].mean():.4f} +/- {leg[:,0].std():.4f}",
                             f"{np.mean([r['f1'] for r in rf_blk]):.4f} +/- {np.std([r['f1'] for r in rf_blk]):.4f}"],
                   "RF_AUC": [f"{leg[:,1].mean():.4f} +/- {leg[:,1].std():.4f}",
                              f"{np.mean([r['auc'] for r in rf_blk]):.4f} +/- {np.std([r['auc'] for r in rf_blk]):.4f}"]})
e6.to_csv(os.path.join(OUT, "E6_leakage_gap.csv"), index=False); log(e6.to_string(index=False))

# ---- E7: detection latency + inference time ----
log("\n=== E7: detection latency (seconds from fire onset to first alarm), pooled over seeds ===")
lrows = []
for k in lat_pool:
    if k == "Majority Class": continue
    l, miss = np.array(lat_pool[k]["lat"]), lat_pool[k]["missed"]
    n = len(l) + miss
    lrows.append(dict(model=k, onset_events=n, detected_pct=100 * len(l) / max(n, 1),
                      median_s=float(np.median(l)) if len(l) else np.nan,
                      p90_s=float(np.percentile(l, 90)) if len(l) else np.nan, missed=miss))
lat_tab = pd.DataFrame(lrows).set_index("model"); lat_tab.to_csv(os.path.join(OUT, "E7_detection_latency.csv"))
log(lat_tab.round(2).to_string())
log("NOTE: onsets are only counted where the 0->1 transition lies inside a test block; "
    "blocks cut some events, so interpret as a lower-bound evaluation.")
S, te = saved["suite"], saved["te"]; Xte = te[ALL_FEATS].values; Xs = S.sc.transform(Xte)
def tm(fn, rep=3):
    best = 1e9
    for _ in range(rep):
        a = time.perf_counter(); fn(); best = min(best, time.perf_counter() - a)
    return 1000 * best / len(Xte)
inf = {"Random Forest": tm(lambda: S.rf.predict_proba(Xte)), "XGBoost": tm(lambda: S.xgb.predict_proba(Xte)),
       "ANN": tm(lambda: S.ann.predict_proba(Xs)), "Isolation Forest": tm(lambda: S._iso(Xs)),
       "Autoencoder": tm(lambda: S._ae(Xs)), "Hybrid Ensemble (all components)": tm(lambda: S.scores(te))}
inf_tab = pd.Series(inf, name="ms_per_sample (batch)"); inf_tab.to_csv(os.path.join(OUT, "E7_inference_time.csv"))
log("Inference time (ms/sample, batch mode, this machine):\n" + inf_tab.round(4).to_string())

# ---- SHAP + figures (seed-0 split) ----
log("\n=== Figures and SHAP ===")
plt.rcParams["font.family"] = "serif"
core = ["Random Forest", "XGBoost", "ANN", "Isolation Forest", "Autoencoder", "Hybrid Ensemble"]
fig, ax = plt.subplots(figsize=(11, 5)); w = .16
for i, mt in enumerate(["accuracy", "precision", "recall", "f1", "auc"]):
    ax.bar(np.arange(len(core)) + (i - 2) * w, [main_tab.loc[m, mt + "_mean"] for m in core],
           w, yerr=[main_tab.loc[m, mt + "_std"] for m in core], label=mt.upper() if mt == "auc" else mt.capitalize(), capsize=2)
ax.set_xticks(range(len(core))); ax.set_xticklabels(core, rotation=15, ha="right"); ax.set_ylim(0, 1.05)
ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.35)); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Block-wise test performance (mean +/- std over split seeds)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "model_comparison.png"), dpi=300, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 6))
for m in core:
    fp, tp, _ = roc_curve(te["y"], saved["sc"][m]); ax.plot(fp, tp, label=f"{m} (AUC={roc_auc_score(te['y'], saved['sc'][m]):.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(frameon=False, fontsize=8)
ax.set_title("ROC (block-wise test split, seed 42)"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "roc_curves.png"), dpi=300); plt.close()

for m in core:
    cm = confusion_matrix(te["y"], saved["pred"][m], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.8)); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["No Fire", "Fire"]); ax.set_yticklabels(["No Fire", "Fire"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(m, fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"confusion_{m.replace(' ', '_')}.png"), dpi=300); plt.close()

try:
    import shap
    idx = np.random.default_rng(0).choice(len(te), size=min(500, len(te)), replace=False)
    Xs_df = te[ALL_FEATS].iloc[idx]
    sv = np.array(shap.TreeExplainer(S.rf).shap_values(Xs_df, check_additivity=False))
    if sv.ndim == 3: sv = sv[:, :, 1]
    imp = sorted(zip(ALL_FEATS, np.abs(sv).mean(0).tolist()), key=lambda x: -x[1])
    pd.DataFrame(imp, columns=["feature", "mean_abs_shap"]).to_csv(os.path.join(OUT, "shap_importance.csv"), index=False)
    log("Top-10 SHAP (RF, test sample):"); [log(f"  {a:22s}{b:.4f}") for a, b in imp[:10]]
    plt.figure(); shap.summary_plot(sv, Xs_df, show=False); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "shap_summary.png"), dpi=300, bbox_inches="tight"); plt.close()
    results["shap_importance"] = imp
except Exception as e:
    log("SHAP skipped:", repr(e))

# ----------------------------- save everything ---------------------------
results["meta"] = dict(seeds=SEEDS, block_rows=BLOCK, cv_folds=CV_FOLDS, cv_repeats=CV_REPEATS, n_rows=len(df),
                       fire_prevalence=float(df["y"].mean()), features=ALL_FEATS, quick=QUICK,
                       single_feature_chosen=saved["suite"].sf_name, runtime_s=time.time() - t0)
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
log(f"\nDONE in {time.time() - t0:.0f}s. Everything is in {os.path.abspath(OUT)}")
log("Send back: run_log.txt and the CSV files (E1..E7), results.json.")
