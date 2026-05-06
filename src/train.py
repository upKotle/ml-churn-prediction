from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.preprocessing import load_and_prepare_data


def train_model(data_path):
    X, y = load_and_prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)

    return score


if __name__ == "__main__":
    score = train_model("data/telco_churn.csv")
    print(f"ROC-AUC: {score:.3f}")
