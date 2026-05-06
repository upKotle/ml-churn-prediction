import pandas as pd
import pytest

from src.preprocessing import validate_raw_data


def test_validate_raw_data_rejects_empty_dataframe():
    empty_frame = pd.DataFrame(columns=["customerID", "Churn", "TotalCharges"])

    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(empty_frame)


def test_validate_raw_data_rejects_invalid_churn_values(sample_raw_dataframe):
    sample_raw_dataframe.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Invalid churn values"):
        validate_raw_data(sample_raw_dataframe)