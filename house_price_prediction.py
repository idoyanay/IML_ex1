

import pandas as pd
import numpy as np
from typing import Tuple, NoReturn


def preprocess_train(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    
    # - drop line with to much bedrooms
    # - try to remove/un-remove the line of sqrt_living15 and sqrt_lot15
    # - try to remove/un-remove the coloumn of 'waterfront'
    # Create a copy to avoid modifying the original data

    X_clean = X.copy()
    y_clean = y.copy()
    
    # Example: Remove rows where the house was built after renovation (illogical)
    invalid_years = X_clean["yr_built"] > X_clean["yr_renovated"]
    X_clean, y_clean = X_clean[~invalid_years], y_clean[~invalid_years]
    
    X_clean, y_clean = remove_small_living_area(X_clean, y_clean)
    
    # Fill missing numeric values with median
    num_cols = X_clean.select_dtypes(include=[np.number]).columns
    X_clean[num_cols] = X_clean[num_cols].fillna(X_clean[num_cols].median())

    # TODO: check if there are any unwanted negative values
    check_negative_values(X_clean, y_clean)

    # Handle houses sold twice by taking the average of their features and target
    X_clean, y_clean = avarge_house_prices(X_clean, y_clean)

    
    


    return X_clean, y_clean


def avarge_house_prices(X_clean: pd.DataFrame, y_clean: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle houses sold twice by taking the average of their features and target
    Parameters
    ----------
    X_clean: pd.DataFrame
        the loaded data
        y_clean: pd.Series
        the loaded data
    Returns
    -------
    A clean, preprocessed version of the data
    1. Identify duplicate houses by index (e.g., id)
    2. Get all duplicate rows
    3. Average only the target (price)
    4. Keep only the first occurrence of each duplicate group for features
    5. Drop original duplicates from X and y
    6. Append the cleaned version back
    7. Optional: sort
    8. Return the cleaned X and y
    """
    # Step 1: Identify duplicate houses by index (e.g., id)
    duplicate_indices = X_clean.index[X_clean.index.duplicated(keep=False)]

    if not duplicate_indices.empty:
        # Step 2: Get all duplicate rows
        duplicates_X = X_clean.loc[duplicate_indices]
        duplicates_y = y_clean.loc[duplicate_indices]

        # Step 3: Average only the target (price)
        avg_target = duplicates_y.groupby(duplicates_y.index).mean()

        # Step 4: Keep only the first occurrence of each duplicate group for features
        first_features = duplicates_X[~duplicates_X.index.duplicated(keep='first')]

        # Step 5: Drop original duplicates from X and y
        X_clean = X_clean.drop(duplicate_indices)
        y_clean = y_clean.drop(duplicate_indices)

        # Step 6: Append the cleaned version back
        X_clean = pd.concat([X_clean, first_features])
        y_clean = pd.concat([y_clean, avg_target])

        # Optional: sort
        X_clean = X_clean.sort_index()
        y_clean = y_clean.sort_index()

    return X_clean, y_clean

def check_negative_values(X_clean: pd.DataFrame, y_clean: pd.Series) -> NoReturn:
        # Step 1: Exclude 'lat' and 'long' from the check
    columns_to_check = [col for col in X_clean.columns if col not in ['lat', 'long']]

    # Step 2: Keep only numeric columns from the rest
    numeric_cols = X_clean[columns_to_check].select_dtypes(include=[np.number]).columns

    # Step 3: Find columns with any negative values
    negative_cols = [col for col in numeric_cols if (X_clean[col] < 0).any()]

    print("Columns with negative values (excluding 'lat' and 'long'):")
    print(negative_cols)

    # Step 4: Extract rows where any of those columns have a negative value
    rows_with_negatives = X_clean[X_clean[negative_cols].lt(0).any(axis=1)]

    print(f"\nNumber of rows with negative values: {len(rows_with_negatives)}")
    print(rows_with_negatives)

def remove_small_living_area(X_clean: pd.DataFrame, y_clean: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows with unrealistically small living room size
    Parameters
    ----------
    X_clean: pd.DataFrame
        the loaded data
    y_clean: pd.Series
        the loaded data
    Returns
    -------
    A clean, preprocessed version of the data
    """
    # Example: Remove rows with very small living room size (e.g., < 10 m²)
    base_area = 30  # reduced to allow more houses to pass the filter
    bathroom_area = 5
    min_bedroom_area = 5  # minimum allowed average size per bedroom (in m²)

    # Step 1: Remove lines without bedrooms or bathrooms
    X_clean = X_clean[(X_clean["bedrooms"] > 0) & (X_clean["bathrooms"] > 0)]
    y_clean = y_clean[X_clean.index]  # sync y with X  

    # Step 2: Compute average bedroom area
    avg_room_area = (
        X_clean["sqft_living"]
        - (X_clean["bathrooms"] * bathroom_area)
        - base_area
    ) / X_clean["bedrooms"]

    # Step 3: Remove houses with unrealistically small bedrooms
    small_living = avg_room_area < min_bedroom_area
    X_clean = X_clean[~small_living]
    y_clean = y_clean[~small_living]
    return X_clean, y_clean

        
def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    return


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pass



def Q2(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into train and test sets. The test set should be 20% of the data.
    Parameters
    ----------
    X: pd.DataFrame
        Design matrix of regression problem

    y: pd.Series
        Response vector to split

    seed: int
        Random seed for reproducibility
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Get the total number of samples
    n_samples = X.shape[0]

    # Create a shuffled array of indices
    indices = np.random.permutation(n_samples)

    # Find the split point
    split_point = int(0.75 * n_samples)

    # Split the indices
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    # Use the indices to split X and y
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

import plotly.express as px

def plot_training_data_plotly(X_train, y_train):
    # Combine features and labels into one DataFrame
    df_plot = X_train.copy()
    df_plot["target"] = y_train.values

    # Check that there are at least two features
    if X_train.shape[1] < 2:
        raise ValueError("Need at least two features to plot.")

    # Pick first two features
    x_col = X_train.columns[0]
    y_col = X_train.columns[1]

    fig = px.scatter(df_plot, x=x_col, y=y_col, color=df_plot["target"].astype(str),
                     title="Training Set (Colored by Target)",
                     labels={"color": "Target"},
                     symbol="target")
    fig.show()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price
    X = X.set_index("id")
    y.index = X.index  # keep them in sync
    # Optional: sort    
    X = X.sort_index()
    y = y.sort_index()




    # Question 2 - split train test
    X_train, X_test, y_train, y_test = Q2(X, y)
    plot_training_data_plotly(X_train, y_train)
    

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # notes:


    # Question 4 - Feature evaluation of train dataset with respect to response

    # Question 5 - preprocess the test data

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

