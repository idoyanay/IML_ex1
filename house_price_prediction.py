

import pandas as pd
import numpy as np
from typing import Tuple


def preprocess_train(X: pd.DataFrame, y: pd.Series):
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
    pass


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


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price


    # Question 2 - split train test
    train_df, test_df = Q2(X, y)



    
    

    # Question 3 - preprocessing of housing prices train dataset

    # Question 4 - Feature evaluation of train dataset with respect to response

    # Question 5 - preprocess the test data

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

