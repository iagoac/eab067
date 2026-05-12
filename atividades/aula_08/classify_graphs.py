"""
classify_graphs.py
------------------
Reads graph_metrics.csv and trains a Random Forest classifier to predict
whether a graph is dense (is_dense = True/False).

Every cross-validation strategy presented in the scikit-learn docs is exercised:
https://scikit-learn.org/stable/modules/cross_validation.html

Results are exported to cv_results.csv.
"""

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    RepeatedKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    PredefinedSplit,
    GroupKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GroupShuffleSplit,
    TimeSeriesSplit,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------

df = pd.read_csv("graph_metrics.csv")

# Coerce columns that may contain "nan" strings
numeric_cols = ["avg_path_length", "small_world_sigma", "modularity"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Features — exclude density to avoid trivial leakage (is_dense := density > 0.5)
FEATURE_COLS = [
    "nodes",
    "edges",
    "avg_clustering_coefficient",
    "avg_betweenness_centrality",
    "avg_degree_centrality",
    "avg_path_length",
    "small_world_sigma",
    "modularity",
]

df_clean = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

X = df_clean[FEATURE_COLS].values
y = df_clean["is_dense"].astype(int).values  # 1 = dense, 0 = sparse

# Groups: integer-encoded graph_class (used by group-based splitters)
le = LabelEncoder()
groups = le.fit_transform(df_clean["graph_class"].values)
n_groups = len(le.classes_)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_groups} graph classes")
print(f"Class balance — dense: {y.sum()}, sparse: {(y == 0).sum()}\n")

# ---------------------------------------------------------------------------
# 2. Classifier
# ---------------------------------------------------------------------------

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# ---------------------------------------------------------------------------
# 3. Cross-validation strategies
# ---------------------------------------------------------------------------
# Each entry:
#   name        – human-readable label
#   cv          – CV splitter object
#   needs_groups – whether to pass groups= to cross_val_score
#   note        – optional remark recorded in the output
#
# For LeavePOut(p=2) the number of splits is C(n,2) ≈ 500 K on 1000 rows,
# which is computationally intractable.  We therefore run it on a 100-row
# subset and record that fact in the note column.

N = len(X)

# PredefinedSplit: last 20 % of rows are the validation fold (test_fold=0),
# the rest are always in the training set (test_fold=-1).
predefined_folds = np.full(N, -1)
predefined_folds[int(0.8 * N):] = 0

CV_STRATEGIES = [
    # ── i.i.d. iterators ────────────────────────────────────────────────────
    {
        "name": "KFold (k=5)",
        "cv": KFold(n_splits=5, shuffle=True, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    {
        "name": "KFold (k=10)",
        "cv": KFold(n_splits=10, shuffle=True, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    {
        "name": "RepeatedKFold (k=5, repeats=3)",
        "cv": RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    {
        "name": "LeaveOneOut",
        "cv": LeaveOneOut(),
        "needs_groups": False,
        "note": f"n_splits={N}",
    },
    {
        "name": "LeavePOut (p=2, subset n=100)",
        "cv": LeavePOut(p=2),
        "needs_groups": False,
        "note": "Run on first 100 rows; C(n,2) is intractable on full dataset",
        "subset": 100,  # special key: use only this many rows
    },
    {
        "name": "ShuffleSplit (splits=10, test=0.2)",
        "cv": ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    # ── stratified iterators ─────────────────────────────────────────────────
    {
        "name": "StratifiedKFold (k=5)",
        "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    {
        "name": "StratifiedKFold (k=10)",
        "cv": StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    {
        "name": "RepeatedStratifiedKFold (k=5, repeats=3)",
        "cv": RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    {
        "name": "StratifiedShuffleSplit (splits=10, test=0.2)",
        "cv": StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
        "needs_groups": False,
        "note": "",
    },
    # ── predefined split ─────────────────────────────────────────────────────
    {
        "name": "PredefinedSplit (last 20% as val fold)",
        "cv": PredefinedSplit(test_fold=predefined_folds),
        "needs_groups": False,
        "note": "test_fold=0 for last 20%, -1 (always train) for the rest",
    },
    # ── group-aware iterators ────────────────────────────────────────────────
    {
        "name": f"GroupKFold (k={n_groups})",
        "cv": GroupKFold(n_splits=n_groups),
        "needs_groups": True,
        "note": f"Groups = graph_class ({n_groups} unique classes)",
    },
    {
        "name": f"StratifiedGroupKFold (k={min(n_groups, 5)})",
        "cv": StratifiedGroupKFold(n_splits=min(n_groups, 5), shuffle=True, random_state=42),
        "needs_groups": True,
        "note": f"Groups = graph_class",
    },
    {
        "name": "LeaveOneGroupOut",
        "cv": LeaveOneGroupOut(),
        "needs_groups": True,
        "note": f"n_splits = n_groups = {n_groups}",
    },
    {
        "name": "LeavePGroupsOut (p=2)",
        "cv": LeavePGroupsOut(n_groups=2),
        "needs_groups": True,
        "note": f"C({n_groups},2) = {n_groups*(n_groups-1)//2} splits",
    },
    {
        "name": "GroupShuffleSplit (splits=10, test=0.2)",
        "cv": GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
        "needs_groups": True,
        "note": f"Groups = graph_class",
    },
]

# ---------------------------------------------------------------------------
# 4. Run experiments
# ---------------------------------------------------------------------------

SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]

records = []

for strategy in CV_STRATEGIES:
    name = strategy["name"]
    cv = strategy["cv"]
    needs_groups = strategy["needs_groups"]
    note = strategy["note"]
    subset = strategy.get("subset", None)

    if subset is not None:
        X_cv, y_cv, g_cv = X[:subset], y[:subset], groups[:subset]
    else:
        X_cv, y_cv, g_cv = X, y, groups

    cv_groups = g_cv if needs_groups else None

    print(f"Running: {name} ...", end=" ", flush=True)

    row = {"cv_strategy": name, "note": note}
    t_start = time.perf_counter()

    for metric in SCORING:
        try:
            scores = cross_val_score(
                clf, X_cv, y_cv,
                cv=cv,
                scoring=metric,
                groups=cv_groups,
                error_score="raise",
            )
            row[f"{metric}_mean"] = round(float(scores.mean()), 6)
            row[f"{metric}_std"] = round(float(scores.std()), 6)
            row[f"{metric}_n_splits"] = len(scores)
        except Exception as exc:
            row[f"{metric}_mean"] = "error"
            row[f"{metric}_std"] = "error"
            row[f"{metric}_n_splits"] = "error"
            print(f"\n  [WARNING] {metric}: {exc}", end=" ")

    row["elapsed_seconds"] = round(time.perf_counter() - t_start, 4)
    records.append(row)
    print(f"done ({row['elapsed_seconds']:.2f}s)")

# ---------------------------------------------------------------------------
# 5. Export results
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(records)
results_df.to_csv("cv_results.csv", index=False)
print(f"\nResults exported to 'cv_results.csv' ({len(results_df)} strategies × {len(results_df.columns)} columns)")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------

print("\n=== Accuracy summary (mean ± std) ===")
summary_cols = ["cv_strategy", "accuracy_mean", "accuracy_std", "accuracy_n_splits"]
available = [c for c in summary_cols if c in results_df.columns]
print(results_df[available].to_string(index=False))
