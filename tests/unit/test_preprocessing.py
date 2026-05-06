import pandas as pd
import pytest

from src.preprocessing import load_and_prepare_data, prepare_features, validate_raw_data


@pytest.mark.smoke
def test_validate_raw_data_accepts_required_columns(sample_raw_dataframe):
    validate_raw_data(sample_raw_dataframe)


@pytest.mark.parametrize("missing_column", ["customerID", "Churn", "TotalCharges"])
def test_validate_raw_data_rejects_missing_required_columns(sample_raw_dataframe, missing_column):
    df = sample_raw_dataframe.drop(columns=[missing_column])

    with pytest.raises(ValueError, match=missing_column):
        validate_raw_data(df)


def test_prepare_features_converts_target_and_total_charges(sample_raw_dataframe):
    features, target = prepare_features(sample_raw_dataframe)

    assert list(target) == [0, 1, 0]
    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert features["TotalCharges"].tolist() == [29.85, 0.0, 240.0]
    assert features["TotalCharges"].dtype.kind in {"f", "i"}


def test_prepare_features_one_hot_encodes_categorical_columns(sample_raw_dataframe):
    features, _ = prepare_features(sample_raw_dataframe)

    expected_columns = {
        "gender_Male",
        "Partner_Yes",
        "Dependents_Yes",
        "PhoneService_Yes",
        "MultipleLines_No phone service",
        "InternetService_Fiber optic",
        "InternetService_No",
        "Contract_Two year",
        "Contract_One year",
        "PaymentMethod_Mailed check",
    }

    assert expected_columns.issubset(features.columns)


def test_load_and_prepare_data_reads_csv(sample_csv_path):
    features, target = load_and_prepare_data(sample_csv_path)

    assert features.shape[0] == 3
    assert target.tolist() == [0, 1, 0]
    assert isinstance(features, pd.DataFrame)