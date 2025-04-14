import matplotlib.pyplot as plt
import pandas as pd
import argparse

import numpy as np
from typing import Tuple, NoReturn
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="zipcode-lookup")
import time


### ====================================== ###
### ========== helper functions ========== ###
### ====================================== ###

def convert_data_to_months(X_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'date' column in the DataFrame to a new feature representing months since the first sale.

    Parameters
    ----------
    X_clean : pd.DataFrame
        The input DataFrame containing a 'date' column.

    Returns
    -------
    pd.DataFrame
        A modified DataFrame with a new 'months_since_first_sale' column and the original 'date' column removed.
    """
    X_clean["date"] = pd.to_datetime(X_clean["date"], format="%Y%m%dT%H%M%S", errors="coerce")

    X_clean["months_since_first_sale"] = (
        ((X_clean["date"] - X_clean["date"].min()) / np.timedelta64(1, 'm'))
    )

    # Fill NaNs with the median
    median_months = X_clean["months_since_first_sale"].median()
    X_clean["months_since_first_sale"] = X_clean["months_since_first_sale"].fillna(median_months).astype(int)

    # Drop original string-based date column
    X_clean = X_clean.drop(columns="date")
    return X_clean


def get_zipcode(lat: float, long: float) -> int:
    try:
        location = geolocator.reverse((lat, long), exactly_one=True, timeout=10)
        zipcode = location.raw["address"].get("postcode", None)
        print(f"Fetched zipcode for ({lat}, {long}): {zipcode}")
        time.sleep(1)  # Be respectful to the free service
        return int(zipcode) if zipcode and zipcode.isdigit() else None
    except Exception as e:
        print(f"Failed for ({lat}, {long}): {e}")
        return None


def parse_arguments() -> dict:
    """
    Parses command-line arguments for the house price prediction script.

    If debug mode is enabled, an input file must be provided.

    Returns
    -------
    dict
        Dictionary with:
        - 'debug': whether debug mode is active
        - 'input': input file path (required in debug mode)
    """
    parser = argparse.ArgumentParser(description="House Price Prediction CLI")

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode (requires --input)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to input CSV file (required in debug mode)"
    )

    args = parser.parse_args()

    # Enforce: if debug is active, input is required
    if args.debug and not args.input:
        parser.error("--debug requires --input to be specified")

    return {
        "debug": args.debug,
        "input": args.input
    }




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


def remove_small_living_area(X_clean: pd.DataFrame, y_clean: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows with unrealistically small living room size.

    Parameters
    ----------
    X_clean : pd.DataFrame
        The loaded design matrix.
    y_clean : pd.Series
        The corresponding response vector.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A clean, preprocessed version of the data.
    """
    base_area = 30  # kitchen + living room
    bathroom_area = 5
    min_bedroom_area = 5

    # Step 1: Remove invalid entries
    X_clean = X_clean[(X_clean["bedrooms"] > 0) & (X_clean["bathrooms"] > 0)]
    y_clean = y_clean.loc[X_clean.index]  # ensure index alignment

    # Step 2: Calculate avg room area
    avg_room_area = (
        X_clean["sqft_living"]
        - (X_clean["bathrooms"] * bathroom_area)
        - base_area
    ) / X_clean["bedrooms"]

    # Step 3: Remove rows with small bedrooms
    small_living = avg_room_area < min_bedroom_area
    X_clean = X_clean[~small_living]
    y_clean = y_clean.loc[X_clean.index]

    return X_clean, y_clean


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
    
    # Create a copy to avoid modifying the original data

    X_clean = X.copy()
    y_clean = y.copy()
    
    # Example: Add a zipcode column based on lat/long -- adding this data for trying to evaluate the level of the area

    # add a zipcode column using geopy -- need to be done in preprocess test also -- 
    if ("zipcode" not in X_clean.columns) and ("lat" in X_clean.columns) and ("long" in X_clean.columns):
        print("Adding zipcode column based on lat/long...")
        # Add a new column 'zipcode' to the DataFrame
        X_clean["zipcode"] = X_clean.apply(lambda x: get_zipcode(x["lat"], x["long"]), axis=1)
    


    # Remove rows with NaN index and rows with NaN price
    X_clean = X_clean[~X_clean.index.isna()]
    y_clean = y_clean[~y_clean.index.isna()]
    y_clean = y_clean.dropna()
    X_clean = X_clean.loc[y_clean.index]

    
    # make sure that the sqft_lot15 is positive - one value seems to be negative, probably a typo
    X_clean.loc[X_clean["sqft_lot15"] < 0, "sqft_lot15"] = (
        X_clean.loc[X_clean["sqft_lot15"] < 0, "sqft_lot15"].abs()
    )

    
    # Convert the 'date' column to a new feature representing months since the first sale -- need to be done in preprocess test also --
    X_clean = convert_data_to_months(X_clean)
    


    

    # Set yr_renovated to 0 for houses where yr_built is later than yr_renovated
    X_clean.loc[X_clean["yr_built"] > X_clean["yr_renovated"], "yr_renovated"] = 0

    
    
    # remove houses with unrealistically small living room size 
    X_clean, y_clean = remove_small_living_area(X_clean, y_clean)


    


    ### ---remove unnecessary columns (optional)--- ###
    # Remove the columns 'sqrt_living15' and 'sqrt_lot15'
    # X_clean = X_clean.drop(columns=["sqrt_living15", "sqrt_lot15"], errors="ignore")

    # remove the column 'waterfront'
    # X_clean = X_clean.drop(columns=["waterfront"], errors="ignore")
    ### ----------------------------------- ###



    # Handle houses sold twice by taking the average of their features and target
    X_clean, y_clean = avarge_house_prices(X_clean, y_clean)



    
    


    return X_clean, y_clean





## ====================================== ##
## ========== main function ========== ###
## ====================================== ##


        
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
    X_test = X.copy()
    
    # add a zipcode column using geopy -- need to be done in preprocess test also -- 
    if ("zipcode" not in X_test.columns) and ("lat" in X_test.columns) and ("long" in X_test.columns):
        print("Adding zipcode column based on lat/long...")
        # Add a new column 'zipcode' to the DataFrame
        X_test["zipcode"] = X_test.apply(lambda x: get_zipcode(x["lat"], x["long"]), axis=1)

    # Convert the 'date' column to a new feature representing months since the first sale -- need to be done in preprocess test also --
    X_test = convert_data_to_months(X_test)

    # Set yr_renovated to 0 for houses where yr_built is later than yr_renovated
    X_test.loc[X_test["yr_built"] > X_test["yr_renovated"], "yr_renovated"] = 0
    return X_test


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


    for feature in X.columns:
        # Step 1: Calculate the correlation
        cov_xy = np.cov(X[feature], y, ddof=0)[0, 1]
        std_x = X[feature].std(ddof=0)
        std_y = y.std(ddof=0)

        if std_x == 0 or std_y == 0:
            corr = 0  # or np.nan, depending on what makes sense
        else:
            corr = cov_xy / (std_x * std_y)

        title = f"Feature: {feature}, Correlation: {corr:.4f}"
        # Step 2: Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, alpha=0.5)
        plt.title(title)
        plt.xlabel(feature)
        plt.ylabel("Response (y)")
        plt.grid(True)

        # Step 3: Save the plot
        file_name = f"{feature}_correlation.png"
        plt.savefig(f"{output_path}/{file_name}")
        plt.close()



def split_data(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into train and test sets (75% train, 25% test).
    Ensures that any row with NaN in index or y, or negative 'sqft_lot15',
    is included in the training set for preprocessing.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix of regression problem

    y : pd.Series
        Response vector to split

    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple of:
        - X_train : pd.DataFrame
        - X_test  : pd.DataFrame
        - y_train : pd.Series
        - y_test  : pd.Series
    """
    # Identify rows with NaN in index or y, or negative sqft_lot15
    mask = X.index.isna() | y.isna() | (X["sqft_lot15"] < 0)

    # Force those into the training set
    X_nan = X[mask]
    y_nan = y[mask]

    # Remaining valid rows
    X_valid = X[~mask]
    y_valid = y[~mask]

    # Shuffle and split
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(X_valid.shape[0])
    split_point = int(0.75 * len(shuffled_indices))

    train_idx = shuffled_indices[:split_point]
    test_idx = shuffled_indices[split_point:]

    # Assemble final sets
    X_train = pd.concat([X_valid.iloc[train_idx], X_nan])
    y_train = pd.concat([y_valid.iloc[train_idx], y_nan])

    X_test = X_valid.iloc[test_idx]
    y_test = y_valid.iloc[test_idx]

    return X_train, X_test, y_train, y_test








if __name__ == '__main__':
    args = parse_arguments()

    if args["debug"]:
        csv_path = args["input"]
        print(f"⚠️ Debug mode enabled - using {csv_path}")
    else:
        csv_path = "house_prices.csv"

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded data from {csv_path}")


    X, y = df.drop("price", axis=1), df.price
    X = X.set_index("id")
    y.index = X.index  # keep them in sync

    # sort the dataframes by index
    X = X.sort_index()
    y = y.sort_index()




    # Question 2 - split train test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path="feature_evaluation")


    # Question 5 - preprocess the test data

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)


