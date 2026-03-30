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
    """
    - prepare the data for training the model
    - split the data into training and testing sets
    - train the model
    - return the model
    """
    X = data.drop("target",axis=1)
    Y = data["target"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", round(accuracy,4)*100,'%')
    return model
