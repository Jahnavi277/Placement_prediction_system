import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "ML",
    "plots"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ML",
    "models"
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FEATURE_COL = "CGPA"
TARGET_COL = "Salary Package"

LEARNING_RATE = 0.1
EPOCHS = 200
RANDOM_SEED = 42


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_and_prepare_data(
    csv_path: str,
    feature_col: str,
    target_col: str
):
    """
    Load the cleaned dataset.

    Only placed students with a valid salary are used
    for the CGPA -> Salary regression experiment.
    """

    df = pd.read_csv(csv_path)

    if feature_col not in df.columns:
        raise KeyError(
            f"Column '{feature_col}' not found."
        )

    if target_col not in df.columns:
        raise KeyError(
            f"Column '{target_col}' not found."
        )

    df = df.dropna(
        subset=[feature_col, target_col]
    )

    # Salary = 0 represents unplaced students.
    df = df[df[target_col] > 0]

    x = df[feature_col].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    print(
        f"Loaded {len(df):,} placed students "
        f"with valid '{feature_col}' and '{target_col}'."
    )

    return x, y


# --------------------------------------------------
# Standardization
# --------------------------------------------------

def standardize(x: np.ndarray):

    mean = x.mean()
    std = x.std()

    if std == 0:
        raise ValueError(
            "Feature standard deviation is zero."
        )

    x_scaled = (x - mean) / std

    return x_scaled, mean, std


# --------------------------------------------------
# Mean Squared Error
# --------------------------------------------------

def compute_mse(
    theta0: float,
    theta1: float,
    x: np.ndarray,
    y: np.ndarray
):

    y_hat = theta0 + theta1 * x

    return float(
        np.mean((y_hat - y) ** 2)
    )


# --------------------------------------------------
# Batch Gradient Descent
# --------------------------------------------------

def batch_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    epochs: int
):

    m = len(x)

    theta0 = 0.0
    theta1 = 0.0

    mse_history = []

    for epoch in range(epochs):

        # Prediction
        y_hat = theta0 + theta1 * x

        # Error
        error = y_hat - y

        # Gradients
        grad_theta0 = (
            (2.0 / m) *
            np.sum(error)
        )

        grad_theta1 = (
            (2.0 / m) *
            np.sum(error * x)
        )

        # Parameter update
        theta0 -= alpha * grad_theta0
        theta1 -= alpha * grad_theta1

        # Record MSE
        mse_history.append(
            compute_mse(
                theta0,
                theta1,
                x,
                y
            )
        )

    return (
        theta0,
        theta1,
        mse_history
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    np.random.seed(RANDOM_SEED)

    # 1. Load data
    x_raw, y = load_and_prepare_data(
        CSV_PATH,
        FEATURE_COL,
        TARGET_COL
    )

    # 2. Standardize CGPA
    x_scaled, x_mean, x_std = standardize(
        x_raw
    )

    # 3. Train
    (
        theta0_scaled,
        theta1_scaled,
        mse_history
    ) = batch_gradient_descent(
        x_scaled,
        y,
        alpha=LEARNING_RATE,
        epochs=EPOCHS
    )

    # 4. Convert parameters back to original CGPA
    slope_original = (
        theta1_scaled / x_std
    )

    intercept_original = (
        theta0_scaled
        - theta1_scaled * x_mean / x_std
    )

    final_mse = mse_history[-1]

    # 5. NumPy closed-form sanity check
    slope_check, intercept_check = np.polyfit(
        x_raw,
        y,
        deg=1
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(
        "\n--- Batch Gradient Descent Results ---"
    )

    print(
        f"Epochs               : {EPOCHS}"
    )

    print(
        f"Learning rate        : {LEARNING_RATE}"
    )

    print(
        f"Final MSE            : {final_mse:.4f}"
    )

    print(
        "Learned model "
        "(standardized features): "
        f"y = {theta0_scaled:.4f} "
        f"+ {theta1_scaled:.4f} * x_scaled"
    )

    print(
        "Learned model "
        "(original CGPA units): "
        f"Salary = {intercept_original:.4f} "
        f"+ {slope_original:.4f} * CGPA"
    )

    print(
        "\n[Sanity check] NumPy fit:"
    )

    print(
        f"Salary = {intercept_check:.4f} "
        f"+ {slope_check:.4f} * CGPA"
    )

    # --------------------------------------------------
    # Save parameters
    # --------------------------------------------------

    results = {
        "feature": FEATURE_COL,
        "target": TARGET_COL,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "theta0_scaled": float(theta0_scaled),
        "theta1_scaled": float(theta1_scaled),
        "intercept_original": float(
            intercept_original
        ),
        "slope_original": float(
            slope_original
        ),
        "x_mean": float(x_mean),
        "x_std": float(x_std),
        "final_mse": float(final_mse)
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "batch_gd_salary_results.json"
        ),
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------
    # Plot 1: MSE
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        mse_history
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.title(
        "Batch Gradient Descent: "
        "MSE vs Epoch"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "mse_vs_epoch.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Plot 2: Fitted line
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.scatter(
        x_raw,
        y,
        s=8,
        alpha=0.15
    )

    x_line = np.linspace(
        x_raw.min(),
        x_raw.max(),
        100
    )

    y_line = (
        intercept_original
        + slope_original * x_line
    )

    plt.plot(
        x_line,
        y_line,
        linewidth=2.5
    )

    plt.xlabel("CGPA")
    plt.ylabel("Salary Package (LPA)")

    plt.title(
        "CGPA → Salary Package"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "fitted_line.png"
        ),
        dpi=150
    )

    plt.close()

    print(
        "\n=== Batch Gradient Descent Complete ==="
    )


if __name__ == "__main__":
    main()