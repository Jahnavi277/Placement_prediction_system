import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import joblib

from batch_gradient_descent_linear_regression import (
    load_and_prepare_data,
    standardize,
    batch_gradient_descent,
    CSV_PATH,
    FEATURE_COL,
    TARGET_COL,
    LEARNING_RATE,
    EPOCHS
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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

VALIDATION_SIZE = 0.2
RANDOM_SEED = 42


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def report_metrics(
    name,
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(
        f"\n--- {name}: Validation Metrics ---"
    )

    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    x, y = load_and_prepare_data(
        CSV_PATH,
        FEATURE_COL,
        TARGET_COL
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
        random_state=RANDOM_SEED
    )

    print(
        f"Train size: {len(x_train):,} "
        f"| Validation size: {len(x_val):,}"
    )

    # --------------------------------------------------
    # 3. sklearn Linear Regression
    # --------------------------------------------------

    sk_model = LinearRegression()

    sk_model.fit(
        x_train.reshape(-1, 1),
        y_train
    )

    sk_intercept = sk_model.intercept_
    sk_slope = sk_model.coef_[0]

    y_val_pred_sklearn = sk_model.predict(
        x_val.reshape(-1, 1)
    )

    sklearn_metrics = report_metrics(
        "sklearn LinearRegression",
        y_val,
        y_val_pred_sklearn
    )

    print(
        f"Learned model: "
        f"Salary = {sk_intercept:.4f} "
        f"+ {sk_slope:.4f} * CGPA"
    )

    # --------------------------------------------------
    # 4. Save sklearn salary model
    # --------------------------------------------------

    joblib.dump(
        sk_model,
        os.path.join(
            MODEL_DIR,
            "salary_linear_regression.joblib"
        )
    )

    # --------------------------------------------------
    # 5. NumPy Batch Gradient Descent
    # --------------------------------------------------

    x_train_scaled, x_mean, x_std = standardize(
        x_train
    )

    (
        theta0_scaled,
        theta1_scaled,
        mse_history
    ) = batch_gradient_descent(
        x_train_scaled,
        y_train,
        alpha=LEARNING_RATE,
        epochs=EPOCHS
    )

    gd_slope = (
        theta1_scaled / x_std
    )

    gd_intercept = (
        theta0_scaled
        - theta1_scaled * x_mean / x_std
    )

    y_val_pred_gd = (
        gd_intercept
        + gd_slope * x_val
    )

    gd_metrics = report_metrics(
        "NumPy Batch Gradient Descent",
        y_val,
        y_val_pred_gd
    )

    print(
        f"Learned model: "
        f"Salary = {gd_intercept:.4f} "
        f"+ {gd_slope:.4f} * CGPA"
    )

    # --------------------------------------------------
    # 6. Comparison
    # --------------------------------------------------

    print(
        "\n=== Coefficient Comparison ==="
    )

    print(
        f"{'Parameter':<15}"
        f"{'sklearn':>15}"
        f"{'NumPy GD':>15}"
        f"{'Difference':>15}"
    )

    print(
        f"{'Intercept':<15}"
        f"{sk_intercept:>15.4f}"
        f"{gd_intercept:>15.4f}"
        f"{abs(sk_intercept - gd_intercept):>15.4f}"
    )

    print(
        f"{'Slope':<15}"
        f"{sk_slope:>15.4f}"
        f"{gd_slope:>15.4f}"
        f"{abs(sk_slope - gd_slope):>15.4f}"
    )

    print(
        "\n=== Validation Metric Comparison ==="
    )

    print(
        f"{'Metric':<10}"
        f"{'sklearn':>15}"
        f"{'NumPy GD':>15}"
    )

    for key, label in [
        ("mse", "MSE"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("r2", "R²")
    ]:

        print(
            f"{label:<10}"
            f"{sklearn_metrics[key]:>15.4f}"
            f"{gd_metrics[key]:>15.4f}"
        )

    # --------------------------------------------------
    # 7. Save comparison results
    # --------------------------------------------------

    results = {
        "sklearn": {
            "intercept": float(sk_intercept),
            "slope": float(sk_slope),
            **sklearn_metrics
        },
        "gradient_descent": {
            "intercept": float(gd_intercept),
            "slope": float(gd_slope),
            **gd_metrics
        }
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "salary_regression_comparison.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------
    # 8. Actual vs Predicted
    # --------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_val,
        y_val_pred_sklearn,
        s=12,
        alpha=0.3
    )

    lims = [
        min(
            y_val.min(),
            y_val_pred_sklearn.min()
        ),
        max(
            y_val.max(),
            y_val_pred_sklearn.max()
        )
    ]

    plt.plot(
        lims,
        lims,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Actual Salary Package (LPA)"
    )

    plt.ylabel(
        "Predicted Salary Package (LPA)"
    )

    plt.title(
        "Linear Regression: "
        "Actual vs Predicted"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "actual_vs_predicted.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 9. Residual distribution
    # --------------------------------------------------

    residuals = (
        y_val
        - y_val_pred_sklearn
    )

    plt.figure(figsize=(7, 5))

    plt.hist(
        residuals,
        bins=40,
        alpha=0.75
    )

    plt.axvline(
        0,
        linewidth=2,
        linestyle="--"
    )

    plt.xlabel(
        "Residual (Actual - Predicted)"
    )

    plt.ylabel("Frequency")

    plt.title(
        "Residual Distribution"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "residual_distribution.png"
        ),
        dpi=150
    )

    plt.close()

    print(
        "\n=== sklearn vs Gradient Descent Complete ==="
    )


if __name__ == "__main__":
    main()