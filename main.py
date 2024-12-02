import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def preprocess_data(data_path, target_column):
    """
    Preprocess the data by encoding categorical variables into numerical values.

    Parameters:
    - data_path: Path to the dataset CSV file.
    - target_column: The name of the target column in the dataset.

    Returns:
    - X: Processed feature matrix.
    - y: Target values.
    """
    data = pd.read_csv(data_path)


    X = data.drop(columns=[target_column])
    y = data[target_column]


    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_model(data_path, target_column, model_filename):
    """
    Trains a RandomForestClassifier for binary or multi-class classification.

    Parameters:
    - data_path: Path to the dataset CSV file.
    - target_column: The name of the target column in the dataset.
    - model_filename: Filename to save the trained model.

    Returns:
    - None
    """

    X, y = preprocess_data(data_path, target_column)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print(f"Classification Report for {target_column.capitalize()} Model:")
    print(classification_report(y_test, y_pred))


    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)


print("Training Diabetes Model...")
train_model("C:/Users/mwashe/multiple_disease_prediction/diabetes.csv", "diabetes", "diabetes_model.pkl")

print("\nTraining Heart Disease Model...")
train_model("C:/Users/mwashe/multiple_disease_prediction/heart_disease.csv", "heart_disease", "heart_disease_model.pkl")
