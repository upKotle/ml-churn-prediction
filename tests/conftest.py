from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import load_and_prepare_data


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_path(project_root):
    return project_root / "data" / "telco_churn.csv"


@pytest.fixture(scope="session")
def prepared_data(data_path):
    return load_and_prepare_data(data_path)


@pytest.fixture()
def sample_raw_dataframe():
    return pd.DataFrame(
        [
            {
                "customerID": "0001-A",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            },
            {
                "customerID": "0002-B",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 56.95,
                "TotalCharges": "",
                "Churn": "Yes",
            },
            {
                "customerID": "0003-C",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "Yes",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No phone service",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 20.0,
                "TotalCharges": "240.0",
                "Churn": "No",
            },
        ]
    )


@pytest.fixture()
def sample_csv_path(tmp_path, sample_raw_dataframe):
    path = tmp_path / "sample_telco.csv"
    sample_raw_dataframe.to_csv(path, index=False)
    return path