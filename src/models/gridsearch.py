from sklearn.model_selection import GridSearchCV
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

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_regressor = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())

    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    joblib.dump(best_params, os.path.join(output_path, "best_params.pkl"))
    print(f"Best parameters saved to {os.path.join(output_path, 'best_params.pkl')}")


if __name__ == "__main__":
    main()
