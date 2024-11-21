import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Function to load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Function to preprocess the data (drop unnecessary columns and encode labels)
def preprocess_data(df):
    df = df.drop(columns=['Id'])
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    X = df.drop(columns=['Species'])
    Y = df['Species']
    return X, Y


# Function to split data into training and test sets
def split_data(X, Y, test_size=0.3, random_state=42):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


# Function to standardize the dataset
def standardize_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


# Function to evaluate model metrics
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    mse = mean_squared_error(actual, pred)
    return accuracy, precision, recall, f1, mse


# Function to train the model and return the classifier
def train_model(x_train, y_train, params):
    model = SVC(**params)
    model.fit(x_train, y_train)
    return model


if __name__ == '__main__':
    # Load dataset
    file_path = "dataset/Iris.csv"
    df = load_data(file_path)

    # Preprocess the data
    X, Y = preprocess_data(df)

    # Split the dataset
    x_train, x_test, y_train, y_test = split_data(X, Y)

    # Standardize the data
    x_train, x_test = standardize_data(x_train, x_test)

    # Define best hyperparameters
    best_params = {'C': 8.415316062303356, 'kernel': 'linear', 'gamma': 'auto'}

    # Train the model
    classifier = train_model(x_train, y_train, best_params)

    # Make predictions
    y_test_hat = classifier.predict(x_test)

    # Evaluate metrics
    accuracy, precision, recall, f1, mse = eval_metrics(y_test, y_test_hat)

    # Print metrics
    print("Classifier model")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1 Score: {f1}")
    print(f"  Mean Squared Error (MSE): {mse}")
