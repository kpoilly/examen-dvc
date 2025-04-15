from sklearn.model_selection import train_test_split

import os
import click
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

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)
            print(f"{filename} saved in {output_folderpath}")

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except:
        return None

@click.command()
@click.argument('input_path', type=click.Path(exists=False), required=0)
@click.argument('output_path', type=click.Path(exists=False), required=0)
def main(input_path, output_path):
    input_path = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    output_path = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())
    data = load_data(input_path)

    X_train, X_test, y_train, y_test = train_test_split(data, data['silica_concentrate'], test_size=0.2, random_state=42)
    save_dataframes(X_train, X_test, y_train, y_test, output_path)


if __name__ == "__main__":
    main()
