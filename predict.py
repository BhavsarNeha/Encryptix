import joblib
import pandas as pd

def make_prediction(new_data):
    model = joblib.load('model.pkl')
    prediction = model.predict(new_data)
    return prediction

if __name__ == "__main__":
    new_data = pd.read_csv('data/new_data.csv')
    predictions = make_prediction(new_data)
    print(predictions)
