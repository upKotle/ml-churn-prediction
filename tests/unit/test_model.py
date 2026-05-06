import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def build_logistic_model():
    return LogisticRegression(max_iter=1000, solver="liblinear")


@pytest.fixture()
def fitted_model(prepared_data):
    features, target = prepared_data
    model = build_logistic_model()
    model.fit(features, target)
    return model, features, target


def test_logistic_regression_predict_returns_binary_labels(fitted_model):
    model, features, _ = fitted_model

    predictions = model.predict(features.head(10))

    assert set(np.unique(predictions)).issubset({0, 1})
    assert len(predictions) == 10


def test_logistic_regression_predict_proba_returns_probabilities(fitted_model):
    model, features, _ = fitted_model

    probabilities = model.predict_proba(features.head(7))

    assert probabilities.shape == (7, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.slow
def test_logistic_regression_roc_auc_is_reasonable(data_path):
    from src.preprocessing import load_and_prepare_data

    features, target = load_and_prepare_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_logistic_model()
    model.fit(x_train, y_train)

    score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    assert 0.75 <= score <= 1.0