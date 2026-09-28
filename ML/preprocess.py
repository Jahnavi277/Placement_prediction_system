import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "Data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ML",
    "models"
)

CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "placement_predict_cleaned.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(CLEANED_DATA_PATH)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    df,
    target_column,
    drop_columns=None,
    stratify=False
):

    if drop_columns is None:
        drop_columns = []

    X = df.drop(
        columns=drop_columns + [target_column]
    )

    y = df[target_column]

    stratify_value = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# IDENTIFY FEATURES
# ============================================================

def identify_features(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return (
        numerical_features,
        categorical_features
    )


# ============================================================
# MISSING VALUES
# ============================================================

def handle_missing_values(
    X_train,
    X_test,
    numerical_features
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit ONLY on training data
    X_train[numerical_features] = (
        imputer.fit_transform(
            X_train[numerical_features]
        )
    )

    # Use training statistics on test data
    X_test[numerical_features] = (
        imputer.transform(
            X_test[numerical_features]
        )
    )

    return (
        X_train,
        X_test,
        imputer
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_data(
    X_train,
    X_test,
    numerical_features
):

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit ONLY on training data
    X_train[numerical_features] = (
        scaler.fit_transform(
            X_train[numerical_features]
        )
    )

    # Transform test using training statistics
    X_test[numerical_features] = (
        scaler.transform(
            X_test[numerical_features]
        )
    )

    return (
        X_train,
        X_test,
        scaler
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot_encode_data(
    X_train,
    X_test,
    one_hot_features
):

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    train_encoded = encoder.fit_transform(
        X_train[one_hot_features]
    )

    test_encoded = encoder.transform(
        X_test[one_hot_features]
    )

    encoded_columns = (
        encoder.get_feature_names_out(
            one_hot_features
        )
    )

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    X_train = X_train.drop(
        columns=one_hot_features
    )

    X_test = X_test.drop(
        columns=one_hot_features
    )

    X_train = pd.concat(
        [
            X_train,
            train_encoded_df
        ],
        axis=1
    )

    X_test = pd.concat(
        [
            X_test,
            test_encoded_df
        ],
        axis=1
    )

    return (
        X_train,
        X_test,
        encoder
    )


# ============================================================
# ORDINAL ENCODING
# ============================================================

def ordinal_encode_data(
    X_train,
    X_test,
    ordinal_features
):

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    train_encoded = encoder.fit_transform(
        X_train[ordinal_features]
    )

    test_encoded = encoder.transform(
        X_test[ordinal_features]
    )

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=ordinal_features,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=ordinal_features,
        index=X_test.index
    )

    X_train = X_train.drop(
        columns=ordinal_features
    )

    X_test = X_test.drop(
        columns=ordinal_features
    )

    X_train = pd.concat(
        [
            X_train,
            train_encoded_df
        ],
        axis=1
    )

    X_test = pd.concat(
        [
            X_test,
            test_encoded_df
        ],
        axis=1
    )

    return (
        X_train,
        X_test,
        encoder
    )


# ============================================================
# SAVE PREPROCESSING ARTIFACTS
# ============================================================

def save_preprocessing_artifacts(
    imputer,
    scaler,
    one_hot_encoder,
    ordinal_encoder,
    feature_columns,
    one_hot_features,
    ordinal_features
):

    joblib.dump(
        imputer,
        os.path.join(
            MODEL_DIR,
            "imputer.joblib"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            MODEL_DIR,
            "scaler.joblib"
        )
    )

    joblib.dump(
        one_hot_encoder,
        os.path.join(
            MODEL_DIR,
            "one_hot_encoder.joblib"
        )
    )

    joblib.dump(
        ordinal_encoder,
        os.path.join(
            MODEL_DIR,
            "ordinal_encoder.joblib"
        )
    )

    metadata = {
        "feature_columns": feature_columns,
        "one_hot_features": one_hot_features,
        "ordinal_features": ordinal_features
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "preprocessing_metadata.json"
        ),
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )
    print(
        "PlacementPredict Preprocessing"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        "\nOriginal Dataset Shape:"
    )

    print(df.shape)

    # --------------------------------------------------------
    # 2. Target
    # --------------------------------------------------------

    target_column = "PlacementStatus"

    # Don't allow identifiers/leakage into features
    drop_columns = [
        "StudentID",
        "Salary Package"
    ]

    # --------------------------------------------------------
    # 3. Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        df,
        target_column,
        drop_columns=drop_columns,
        stratify=True
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
    # 4. Identify features
    # --------------------------------------------------------

    (
        numerical_features,
        categorical_features
    ) = identify_features(X_train)

    print(
        "\nNumerical Features:"
    )

    print(numerical_features)

    print(
        "\nCategorical Features:"
    )

    print(categorical_features)

    # --------------------------------------------------------
    # 5. Faculty-defined categorical groups
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

    # Make sure the lists only contain columns
    # that actually exist.
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
    # 6. Missing values
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
    # 7. Standardization
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
    # 8. One-Hot Encoding
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

    else:

        one_hot_encoder = None

    # --------------------------------------------------------
    # 9. Ordinal Encoding
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

    else:

        ordinal_encoder = None

    # --------------------------------------------------------
    # 10. Add target back for saved datasets
    # --------------------------------------------------------

    X_train_saved = X_train.copy()
    X_test_saved = X_test.copy()

    X_train_saved[target_column] = y_train
    X_test_saved[target_column] = y_test

    train_path = os.path.join(
        DATA_DIR,
        "preprocessed_train.csv"
    )

    test_path = os.path.join(
        DATA_DIR,
        "preprocessed_test.csv"
    )

    X_train_saved.to_csv(
        train_path,
        index=False
    )

    X_test_saved.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # 11. Save preprocessing objects
    # --------------------------------------------------------

    save_preprocessing_artifacts(
        imputer,
        scaler,
        one_hot_encoder,
        ordinal_encoder,
        X_train.columns.tolist(),
        one_hot_features,
        ordinal_features
    )

    # --------------------------------------------------------
    # 12. Final report
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "PREPROCESSING COMPLETED"
    )

    print(
        "=========================================="
    )

    print(
        "\nFinal Training Shape:"
    )

    print(X_train_saved.shape)

    print(
        "\nFinal Testing Shape:"
    )

    print(X_test_saved.shape)

    print(
        "\nSaved:"
    )

    print(
        train_path
    )

    print(
        test_path
    )

    print(
        "\nSaved preprocessing artifacts:"
    )

    print(
        "- imputer.joblib"
    )

    print(
        "- scaler.joblib"
    )

    print(
        "- one_hot_encoder.joblib"
    )

    print(
        "- ordinal_encoder.joblib"
    )

    print(
        "- preprocessing_metadata.json"
    )


if __name__ == "__main__":
    main()