from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os


def main():
    output_path = "models"
    try:
        X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
        y_train = pd.read_csv("data/processed_data/y_train.csv")
    except:
        print("Error: X_train_scaled or y_train not found.")
        exit(1)

    try:
        best_params = joblib.load(os.path.join(output_path, "best_params.pkl"))
    except:
        print("Error: best_params.pkl not found.")
        exit(1)

    rf_regressor = RandomForestRegressor(**best_params, random_state=42)
    rf_regressor.fit(X_train, y_train)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    joblib.dump(rf_regressor, os.path.join(output_path, "model.pkl"))
    print(f"Model saved to {os.path.join(output_path, 'model.pkl')}")


if __name__ == "__main__":
    main()
