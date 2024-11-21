import pytest
import pandas as pd
from train import (
    load_data,
    preprocess_data,
    split_data,
    standardize_data,
    train_model,
    eval_metrics,
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Test the `load_data` function
def test_load_data():
    df = load_data("dataset/Iris.csv")
    assert not df.empty, "Dataset is empty!"
    assert "Id" in df.columns, "Expected column 'Id' not found!"
    assert "Species" in df.columns, "Expected column 'Species' not found!"


# Test the `preprocess_data` function
def test_preprocess_data():
    df = pd.read_csv("dataset/Iris.csv")
    X, Y = preprocess_data(df)
    assert not X.empty, "Features dataset (X) is empty!"
    assert len(Y) > 0, "Labels dataset (Y) is empty!"
    assert "Species" not in X.columns, "Species column should not be in features (X)!"


# Test the `split_data` function
def test_split_data():
    df = pd.read_csv("dataset/Iris.csv")
    X, Y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(X, Y)
    assert len(x_train) > 0, "Training set (x_train) is empty!"
    assert len(x_test) > 0, "Test set (x_test) is empty!"
    assert len(y_train) > 0, "Training labels (y_train) are empty!"
    assert len(y_test) > 0, "Test labels (y_test) are empty!"


# Test the `standardize_data` function
def test_standardize_data():
    df = pd.read_csv("dataset/Iris.csv")
    X, Y = preprocess_data(df)
    x_train, x_test, _, _ = split_data(X, Y)
    x_train_scaled, x_test_scaled = standardize_data(x_train, x_test)

    assert x_train_scaled.shape == x_train.shape, "x_train_scaled shape mismatch!"
    assert x_test_scaled.shape == x_test.shape, "x_test_scaled shape mismatch!"


# Test the `train_model` function
def test_train_model():
    df = pd.read_csv("dataset/Iris.csv")
    X, Y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(X, Y)
    x_train_scaled, _ = standardize_data(x_train, x_test)

    params = {'C': 8.415316062303356, 'kernel': 'linear', 'gamma': 'auto'}
    model = train_model(x_train_scaled, y_train, params)
    assert isinstance(model, SVC), "Model is not an instance of SVC!"
    assert model.kernel == "linear", "Kernel mismatch in trained model!"


# Test the `eval_metrics` function
def test_eval_metrics():
    actual = [0, 1, 1, 0]
    predicted = [0, 1, 0, 0]

    accuracy, precision, recall, f1, mse = eval_metrics(actual, predicted)
    assert accuracy == 0.75, f"Expected accuracy 0.75, got {accuracy}"
    assert precision > 0, "Precision should be greater than 0"
    assert recall > 0, "Recall should be greater than 0"
    assert f1 > 0, "F1 score should be greater than 0"
    assert mse >= 0, "Mean Squared Error should be non-negative!"


if __name__ == "__main__":
    pytest.main()
