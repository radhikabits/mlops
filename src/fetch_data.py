# Import necessary libraries
import os
from sklearn.datasets import fetch_california_housing


def save_housing_data(output_dir: str = "data/raw", filename: str = "housing.csv") -> None:
    """
    Fetches the California housing dataset and saves it as a CSV file.

    Parameters:
    - output_dir (str): Directory where the CSV file will be saved.
    - filename (str): Name of the CSV file.

    Creates the directory if it doesn't exist and writes the dataframe to a CSV.
    """
    # Fetch dataset as a pandas DataFrame
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full path for the output CSV file
    output_path = os.path.join(output_dir, filename)

    # Save the dataset to CSV (without row index)
    df.to_csv(output_path, index=False)

    print(f"Data successfully saved to {output_path}")


if __name__ == "__main__":
    # Entry point of the script
    save_housing_data()
