import pandas as pd


REQUIRED_COLUMNS = {
    "customerID",
    "Churn",
    "TotalCharges",
}


def validate_raw_data(df):
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")

    if df.empty:
        raise ValueError("Input data is empty")

    invalid_churn_values = sorted(set(df["Churn"].dropna().unique()) - {"Yes", "No"})
    if invalid_churn_values:
        invalid_list = ", ".join(map(str, invalid_churn_values))
        raise ValueError(f"Invalid churn values: {invalid_list}")


def prepare_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    validate_raw_data(df)
    return prepare_features(df)
