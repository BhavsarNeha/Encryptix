from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(X_test, y_test):
    model = joblib.load('model.pkl')
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2
