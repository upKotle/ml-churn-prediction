import pandas as pd

def load_and_prepare_data(path):
    df = pd.read_csv(path)

    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y
