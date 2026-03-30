import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def prepare_user_sent_data(features: dict) -> pd.DataFrame:
    #convert the user sent data to a dataframe same as the training data
    FEATURE_COLUMNS = ("study_hours", "attendance", "previous_score")
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    try:
        row = [[float(features[c]) for c in FEATURE_COLUMNS]]
    except (TypeError, ValueError) as e:
        raise ValueError("feature values must be numbers") from e
    X = pd.DataFrame(row, columns=list(FEATURE_COLUMNS))
    print("X: ", X)
    return X

def train_model():
    #fix the file read path
    data = pd.read_csv("data/sample.csv")
    print("Data: ", data.head())
    X = data[["study_hours", "attendance", "previous_score"]]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    """
    - prepare the data for training the model
    - split the data into training and testing sets
    - train the model
    - return the model
    """

    return model
