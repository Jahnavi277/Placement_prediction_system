import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "placement_predict_cleaned.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ML",
    "models"
)

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "ML",
    "plots"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TARGET_COL = "PlacementStatus"

EXCLUDE_FROM_FEATURES = [
    "StudentID",
    "Salary Package"
]

VALIDATION_SIZE = 0.2
RANDOM_SEED = 42


# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------

def load_and_prepare_data(
    csv_path: str
):

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target '{TARGET_COL}' not found."
        )

    # Use numeric features, matching the faculty experiment
    numeric_cols = (
        df
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    feature_cols = [
        col
        for col in numeric_cols
        if col != TARGET_COL
        and col not in EXCLUDE_FROM_FEATURES
    ]

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found."
        )

    print(
        "\nNumeric features used:"
    )

    for feature in feature_cols:
        print("-", feature)

    # Clean rows
    df = df.dropna(
        subset=feature_cols + [TARGET_COL]
    )

    x = df[
        feature_cols
    ].to_numpy(
        dtype=float
    )

    y = df[
        TARGET_COL
    ].to_numpy(
        dtype=int
    )

    print(
        f"\nLoaded {len(df):,} rows."
    )

    print(
        "Placed:",
        int(y.sum())
    )

    print(
        "Not placed:",
        int((1 - y).sum())
    )

    return (
        x,
        y,
        feature_cols
    )


# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------

def train_model(
    x_train,
    x_val,
    y_train,
    y_val,
    label
):

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED
    )

    model.fit(
        x_train,
        y_train
    )

    y_pred = model.predict(
        x_val
    )

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    print(
        f"{label:<20}: "
        f"Validation Accuracy = "
        f"{accuracy:.4f}"
    )

    return (
        model,
        accuracy
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    (
        x,
        y,
        feature_cols
    ) = load_and_prepare_data(
        CSV_PATH
    )

    # --------------------------------------------------
    # 2. Train / validation split
    # --------------------------------------------------

    (
        x_train,
        x_val,
        y_train,
        y_val
    ) = train_test_split(
        x,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(
        f"\nTrain size: {len(x_train):,}"
        f" | Validation size: {len(x_val):,}"
    )

    results = {}

    models = {}
    scalers = {}

    # --------------------------------------------------
    # 3. Unscaled
    # --------------------------------------------------

    (
        model_unscaled,
        acc_unscaled
    ) = train_model(
        x_train,
        x_val,
        y_train,
        y_val,
        "Unscaled"
    )

    results["Unscaled"] = acc_unscaled

    models["Unscaled"] = model_unscaled

    # --------------------------------------------------
    # 4. StandardScaler
    # --------------------------------------------------

    standard_scaler = StandardScaler()

    x_train_std = (
        standard_scaler
        .fit_transform(x_train)
    )

    x_val_std = (
        standard_scaler
        .transform(x_val)
    )

    (
        model_standard,
        acc_standard
    ) = train_model(
        x_train_std,
        x_val_std,
        y_train,
        y_val,
        "StandardScaler"
    )

    results["StandardScaler"] = acc_standard

    models["StandardScaler"] = model_standard

    scalers["StandardScaler"] = (
        standard_scaler
    )

    # --------------------------------------------------
    # 5. MinMaxScaler
    # --------------------------------------------------

    minmax_scaler = MinMaxScaler()

    x_train_mm = (
        minmax_scaler
        .fit_transform(x_train)
    )

    x_val_mm = (
        minmax_scaler
        .transform(x_val)
    )

    (
        model_minmax,
        acc_minmax
    ) = train_model(
        x_train_mm,
        x_val_mm,
        y_train,
        y_val,
        "MinMaxScaler"
    )

    results["MinMaxScaler"] = acc_minmax

    models["MinMaxScaler"] = model_minmax

    scalers["MinMaxScaler"] = (
        minmax_scaler
    )

    # --------------------------------------------------
    # 6. Comparison
    # --------------------------------------------------

    print(
        "\n=== Validation Accuracy Comparison ==="
    )

    print(
        f"{'Scaling':<20}"
        f"{'Accuracy':>12}"
    )

    for name, accuracy in results.items():

        print(
            f"{name:<20}"
            f"{accuracy:>12.4f}"
        )

    # --------------------------------------------------
    # 7. Select model for integration
    # --------------------------------------------------

    selected_name = max(
        results,
        key=results.get
    )

    selected_model = models[
        selected_name
    ]

    selected_scaler = scalers.get(
        selected_name
    )

    print(
        f"\nSelected model configuration: "
        f"{selected_name}"
    )

    # --------------------------------------------------
    # 8. Save selected model
    # --------------------------------------------------

    joblib.dump(
        selected_model,
        os.path.join(
            MODEL_DIR,
            "placement_model.joblib"
        )
    )

    # Save scaler only when one is used
    if selected_scaler is not None:

        joblib.dump(
            selected_scaler,
            os.path.join(
                MODEL_DIR,
                "placement_scaler.joblib"
            )
        )

    # --------------------------------------------------
    # 9. Save feature names
    # --------------------------------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "placement_features.json"
        ),
        "w"
    ) as f:

        json.dump(
            feature_cols,
            f,
            indent=4
        )

    # --------------------------------------------------
    # 10. Save model metadata
    # --------------------------------------------------

    metadata = {
        "target": TARGET_COL,
        "selected_configuration": selected_name,
        "validation_size": VALIDATION_SIZE,
        "random_seed": RANDOM_SEED,
        "accuracies": {
            name: float(value)
            for name, value in results.items()
        },
        "feature_count": len(feature_cols),
        "features": feature_cols
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "placement_model_metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # --------------------------------------------------
    # 11. Accuracy comparison plot
    # --------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    names = list(
        results.keys()
    )

    accuracies = list(
        results.values()
    )

    bars = plt.bar(
        names,
        accuracies
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.title(
        "Logistic Regression: "
        "Scaling Comparison"
    )

    plt.ylim(
        0,
        1.0
    )

    for bar, accuracy in zip(
        bars,
        accuracies
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            accuracy + 0.01,
            f"{accuracy:.4f}",
            ha="center"
        )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "scaling_accuracy_comparison.png"
        ),
        dpi=150
    )

    plt.close()

    print(
        "\n=== Logistic Regression Complete ==="
    )

    print(
        "\nSaved model files:"
    )

    print(
        "- placement_model.joblib"
    )

    if selected_scaler is not None:
        print(
            "- placement_scaler.joblib"
        )

    print(
        "- placement_features.json"
    )

    print(
        "- placement_model_metadata.json"
    )


if __name__ == "__main__":
    main()