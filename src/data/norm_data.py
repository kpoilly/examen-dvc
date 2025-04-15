from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os
import sys
import pandas as pd


def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)
            print(f"{filename} saved in {output_folderpath}")


def main():
    output_path = "data/processed_data"
    try:
        X_train = pd.read_csv(output_path + "/X_train.csv")
        X_test = pd.read_csv(output_path + "/X_test.csv")
    except:
        print("Error: X_train or X_test not found.", file=sys.stderr)
        exit(1)

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.drop(columns=['date'])))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test.drop(columns=['date'])))

    save_dataframes(X_train_scaled, X_test_scaled, output_path)


if __name__ == "__main__":
    main()
