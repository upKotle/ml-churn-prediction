from sklearn.linear_model import LogisticRegression

from src.preprocessing import load_and_prepare_data, prepare_features


def test_preprocessing_benchmark(benchmark, data_path):
    benchmark(load_and_prepare_data, data_path)


def test_feature_engineering_benchmark(benchmark, sample_raw_dataframe):
    benchmark(prepare_features, sample_raw_dataframe)


def test_model_inference_benchmark(benchmark, prepared_data):
    features, target = prepared_data
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(features, target)

    benchmark(model.predict_proba, features)