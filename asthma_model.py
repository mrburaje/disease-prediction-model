import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle


def preprocess_data(data_path, target_columns):
    """
    Preprocess the data by encoding categorical variables into numerical values.

    Parameters:
    - data_path: Path to the dataset CSV file.
    - target_columns: A list of target column names in the dataset.

    Returns:
    - X: Processed feature matrix.
    - y: Target values as a DataFrame.
    """
    data = pd.read_csv(data_path)


    X = data.drop(columns=target_columns)
    y = data[target_columns]


    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_multi_target_model(data_path, target_columns, model_filename):
    """
    Trains a MultiOutputClassifier for multi-label classification.

    Parameters:
    - data_path: Path to the dataset CSV file.
    - target_columns: A list of target column names in the dataset.
    - model_filename: Filename to save the trained model.

    Returns:
    - None
    """

    X, y = preprocess_data(data_path, target_columns)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    base_model = RandomForestClassifier(random_state=42)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print(f"Classification Report for Multi-Target Model ({', '.join(target_columns)}):")
    print(classification_report(y_test, y_pred, target_names=target_columns))


    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)


print("\nTraining Asthma Model...")
train_multi_target_model(
    "C:/Users/mwashe/multiple_disease_prediction/asthma.csv",
    ["Severity_Mild", "Severity_Moderate", "Severity_None"],
    "asthma_multi_target_model.pkl"
)
