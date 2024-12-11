import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def split_data(data, target_column):
    """Split data into features and labels."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X.values, y.values
