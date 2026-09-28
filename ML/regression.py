import os
import json
import joblib

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import (
    load_data,
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ML",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(alpha=1.0),

        "Lasso Regression":
            Lasso(alpha=0.01),

        "Elastic Net":
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5
            )
    }

    return models


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    X_test
):

    return model.predict(
        X_test
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    y_test,
    y_pred
):

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "PlacementPredict Salary Regression"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        "\nOriginal Dataset Shape:"
    )

    print(df.shape)

    # --------------------------------------------------------
    # 2. Train/Test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        df,
        target_column="Salary Package",
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "IsAnomaly"
        ],
        stratify=False
    )

    print(
        "\nTraining Shape:"
    )

    print(X_train.shape)

    print(
        "\nTesting Shape:"
    )

    print(X_test.shape)

    # --------------------------------------------------------
    # 3. Identify features
    # --------------------------------------------------------

    (
        numerical_features,
        categorical_features
    ) = identify_features(
        X_train
    )

    print(
        "\nNumerical Features:"
    )

    print(numerical_features)

    print(
        "\nCategorical Features:"
    )

    print(categorical_features)

    # --------------------------------------------------------
    # 4. Faculty categorical groups
    # --------------------------------------------------------

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    # Only use columns that actually exist
    one_hot_features = [
        col
        for col in one_hot_features
        if col in X_train.columns
    ]

    ordinal_features = [
        col
        for col in ordinal_features
        if col in X_train.columns
    ]

    # --------------------------------------------------------
    # 5. Missing values
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        imputer
    ) = handle_missing_values(
        X_train,
        X_test,
        numerical_features
    )

    print(
        "\nMissing Value Handling Completed."
    )

    # --------------------------------------------------------
    # 6. Standardization
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        scaler
    ) = standardize_data(
        X_train,
        X_test,
        numerical_features
    )

    print(
        "Standardization Completed."
    )

    # --------------------------------------------------------
    # 7. One-Hot Encoding
    # --------------------------------------------------------

    if one_hot_features:

        (
            X_train,
            X_test,
            one_hot_encoder
        ) = one_hot_encode_data(
            X_train,
            X_test,
            one_hot_features
        )

        print(
            "One-Hot Encoding Completed."
        )

    # --------------------------------------------------------
    # 8. Ordinal Encoding
    # --------------------------------------------------------

    if ordinal_features:

        (
            X_train,
            X_test,
            ordinal_encoder
        ) = ordinal_encode_data(
            X_train,
            X_test,
            ordinal_features
        )

        print(
            "Ordinal Encoding Completed."
        )

    # --------------------------------------------------------
    # 9. Train models
    # --------------------------------------------------------

    models = create_models()

    results = {}

    for name, model in models.items():

        print(
            "\n" + "=" * 60
        )

        print(
            "Training:",
            name
        )

        print(
            "=" * 60
        )

        # Train
        model = train_model(
            model,
            X_train,
            y_train
        )

        # Predict
        y_pred = predict(
            model,
            X_test
        )

        # Evaluate
        metrics = evaluate_model(
            y_test,
            y_pred
        )

        results[name] = metrics

        print(
            f"MAE  : {metrics['MAE']:.4f}"
        )

        print(
            f"MSE  : {metrics['MSE']:.4f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.4f}"
        )

        print(
            f"R²   : {metrics['R2']:.4f}"
        )

        # Save model
        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                f"{safe_name}.joblib"
            )
        )

    # --------------------------------------------------------
    # 10. Save regression results
    # --------------------------------------------------------

    results_path = os.path.join(
        MODEL_DIR,
        "regression_results.json"
    )

    with open(
        results_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # 11. Print final comparison
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "REGRESSION RESULTS"
    )

    print(
        "=========================================="
    )

    for name, metrics in results.items():

        print(
            f"\n{name}"
        )

        print(
            f"  MAE  : {metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE : {metrics['RMSE']:.4f}"
        )

        print(
            f"  R²   : {metrics['R2']:.4f}"
        )

    print(
        "\nModels saved to:"
    )

    print(
        MODEL_DIR
    )

    print(
        "\nRegression completed successfully."
    )


if __name__ == "__main__":
    main()