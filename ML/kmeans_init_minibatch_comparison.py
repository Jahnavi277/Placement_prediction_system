import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
OPTIMAL_K = 2
N_RESTARTS = 20

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_cleaned.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ML",
    "models"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "ML",
    "plots"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD + PREPROCESS
# ============================================================

print("=" * 65)
print("K-MEANS INITIALIZATION + MINIBATCH COMPARISON")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

EXCLUDE_COLUMNS = [
    "StudentID",
    "PlacementStatus",
    "Salary Package"
]

feature_cols = [
    col for col in numeric_cols
    if col not in EXCLUDE_COLUMNS
]

X_raw = df[feature_cols].copy()


# Median imputation
imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X_raw)


# Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

print(
    f"\nDataset: {X_scaled.shape[0]:,} rows "
    f"x {X_scaled.shape[1]} features"
)

print(
    f"Using K = {OPTIMAL_K}"
)


# ============================================================
# 2. K-MEANS++ VS RANDOM
# ============================================================

print("\n" + "=" * 65)
print(
    f"INITIALIZATION COMPARISON "
    f"({N_RESTARTS} RESTARTS EACH)"
)
print("=" * 65)

inertias = {
    "k-means++": [],
    "random": []
}

times = {
    "k-means++": [],
    "random": []
}


for init_strategy in [
    "k-means++",
    "random"
]:

    print(
        f"\nRunning {init_strategy}..."
    )

    for restart in range(N_RESTARTS):

        model = KMeans(
            n_clusters=OPTIMAL_K,
            init=init_strategy,
            n_init=1,
            random_state=restart
        )

        start = time.perf_counter()

        model.fit(X_scaled)

        elapsed = (
            time.perf_counter() - start
        )

        inertias[init_strategy].append(
            model.inertia_
        )

        times[init_strategy].append(
            elapsed
        )


    values = np.array(
        inertias[init_strategy]
    )

    print(
        f"Mean inertia : {values.mean():,.1f}"
    )

    print(
        f"Std          : {values.std():,.1f}"
    )

    print(
        f"Minimum      : {values.min():,.1f}"
    )

    print(
        f"Maximum      : {values.max():,.1f}"
    )

    print(
        f"Mean time    : "
        f"{np.mean(times[init_strategy]) * 1000:.1f} ms"
    )


# ============================================================
# 3. SAVE RESTART RESULTS
# ============================================================

restart_results = pd.DataFrame({
    "kmeans++_inertia": inertias["k-means++"],
    "random_inertia": inertias["random"]
})

restart_path = os.path.join(
    MODEL_DIR,
    "init_comparison_restarts.csv"
)

restart_results.to_csv(
    restart_path,
    index=False
)


# ============================================================
# 4. BOXPLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.boxplot(
    [
        inertias["k-means++"],
        inertias["random"]
    ],
    tick_labels=[
        "k-means++",
        "random"
    ]
)

ax.set_title(
    f"Inertia Distribution over "
    f"{N_RESTARTS} Restarts (K={OPTIMAL_K})"
)

ax.set_ylabel(
    "Inertia"
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

boxplot_path = os.path.join(
    PLOT_DIR,
    "init_comparison_boxplot.png"
)

plt.savefig(
    boxplot_path,
    dpi=150
)

plt.close()


# ============================================================
# 5. STANDARD K-MEANS
# ============================================================

print("\n" + "=" * 65)
print("STANDARD K-MEANS VS MINIBATCH K-MEANS")
print("=" * 65)

start = time.perf_counter()

standard_kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE
)

standard_kmeans.fit(
    X_scaled
)

standard_time = (
    time.perf_counter() - start
)


# ============================================================
# 6. MINIBATCH K-MEANS
# ============================================================

start = time.perf_counter()

minibatch_kmeans = MiniBatchKMeans(
    n_clusters=OPTIMAL_K,
    init="k-means++",
    n_init=10,
    batch_size=1024,
    random_state=RANDOM_STATE
)

minibatch_kmeans.fit(
    X_scaled
)

minibatch_time = (
    time.perf_counter() - start
)


# ============================================================
# 7. COMPARISON
# ============================================================

standard_inertia = (
    standard_kmeans.inertia_
)

minibatch_inertia = (
    minibatch_kmeans.inertia_
)

speedup = (
    standard_time /
    minibatch_time
)

inertia_difference = (
    (minibatch_inertia - standard_inertia)
    / standard_inertia
    * 100
)

print(
    f"\n{'Method':<22}"
    f"{'Inertia':>15}"
    f"{'Time':>15}"
)

print(
    f"{'Standard KMeans':<22}"
    f"{standard_inertia:>15,.1f}"
    f"{standard_time:>14.3f}s"
)

print(
    f"{'MiniBatch KMeans':<22}"
    f"{minibatch_inertia:>15,.1f}"
    f"{minibatch_time:>14.3f}s"
)

print(
    f"\nMiniBatch speedup: "
    f"{speedup:.1f}x"
)

print(
    f"Inertia difference: "
    f"{inertia_difference:+.2f}%"
)


# ============================================================
# 8. SAVE MINIBATCH COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "Method": "Standard KMeans",
        "Inertia": standard_inertia,
        "Time_sec": standard_time
    },
    {
        "Method": "MiniBatchKMeans",
        "Inertia": minibatch_inertia,
        "Time_sec": minibatch_time
    }
])

comparison_path = os.path.join(
    MODEL_DIR,
    "minibatch_vs_standard.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 9. PLOT FINAL COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 5)
)

methods = [
    "Standard\nKMeans",
    "MiniBatch\nKMeans"
]


# Inertia
axes[0].bar(
    methods,
    [
        standard_inertia,
        minibatch_inertia
    ]
)

axes[0].set_title(
    "Final Inertia"
)

axes[0].set_ylabel(
    "Inertia"
)

axes[0].grid(
    axis="y",
    alpha=0.3
)


# Time
axes[1].bar(
    methods,
    [
        standard_time,
        minibatch_time
    ]
)

axes[1].set_title(
    "Wall-Clock Fit Time"
)

axes[1].set_ylabel(
    "Seconds"
)

axes[1].grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()

comparison_plot_path = os.path.join(
    PLOT_DIR,
    "minibatch_vs_standard.png"
)

plt.savefig(
    comparison_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETE")
print("=" * 65)

print(
    f"Restart results : {restart_path}"
)

print(
    f"Boxplot         : {boxplot_path}"
)

print(
    f"Comparison data : {comparison_path}"
)

print(
    f"Comparison plot : {comparison_plot_path}"
)