from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd

def evaluate_model(X_test, y_test):
    model = joblib.load('model.pkl')
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

if __name__ == "__main__":
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    mse, r2 = evaluate_model(X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
