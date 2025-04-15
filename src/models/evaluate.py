from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import os
import json


def main():
    output_path = "metrics"
    try:
        X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
        y_test = pd.read_csv("data/processed_data/y_test.csv")
        model = joblib.load("models/model.pkl")
    except:
        print("Error: Mandatory files not found.")
        exit(1)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    scores = {
        "mse": mse,
        "r2": r2
    }
    with open(os.path.join(output_path, "scores.json"), "w") as f:
        json.dump(scores, f)
    print(f"Scores saved to {os.path.join(output_path, 'scores.json')}")

    y_pred_df = pd.DataFrame(y_pred, columns=['predicted_silica_concentrate'])
    y_pred_df.to_csv(output_path + "/y_pred.csv", index=False)
    print(f"Predictions saved to data/predicted_data/y_pred.csv")


if __name__ == "__main__":
    main()
