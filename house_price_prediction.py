import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import multiprocessing as mp
import plotly.graph_objects as go
import numpy as np
from typing import Tuple, NoReturn
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="zipcode-lookup") 
from sklearn.neighbors import NearestNeighbors

import time
from linear_regression import LinearRegression
### ====================================== ###
### ========== helper functions ========== ###
### ====================================== ###


price_center = {} # global variable to store the distance from the price center

def update_price_center(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Calculate the price center based on latitude and longitude (weighted average).
    
    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing house features.
    y : pd.Series
        The target variable (house prices).
    """
    global price_center
    if "lat" not in X.columns or "long" not in X.columns:
        raise ValueError("DataFrame must contain 'lat' and 'long' columns.")

    # Calculate the weighted average of latitude and longitude based on house prices
    price_center["lat"] = np.average(X["lat"], weights=y)
    price_center["long"] = np.average(X["long"], weights=y)

    

def create_l2_dist_from_price_center(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature 'l2_dist_from_price_center' based on the L2 geographic distance (based on lat and long) from the price center.
    
    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing house features.
    
    Returns
    -------
    pd.DataFrame
        A modified DataFrame with the new 'l2_dist_from_price_center' feature.
    """
    if "lat" not in X.columns or "long" not in X.columns:
        raise ValueError("DataFrame must contain 'lat' and 'long' columns.")

    # Compute the L2 distance from the price center
    X["l2_dist_from_price_center"] = np.sqrt(
        (X["lat"] - price_center["lat"]) ** 2 + (X["long"] - price_center["long"]) ** 2)

    return X
    
    

def create_closest15_feature(X: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Create a new feature '<feature>_closest15' based on the average of the 15 closest houses for the specified feature.
    
    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing house features.
    feature : str
        The name of the feature to calculate the closest 15 average for. 
    
    Returns
    -------
    pd.DataFrame
        A modified DataFrame with the new '<feature>_closest15' feature.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")
    
    # Create a copy to avoid modifying the original data
    X_copy = X.copy()
    if X_copy.empty:
        raise ValueError("create_closest15_feature received an empty DataFrame.")

    # Fit NearestNeighbors on the data
    knn = NearestNeighbors(n_neighbors=15)
    knn.fit(X_copy[["lat", "long"]])  # Use geographical coordinates for proximity
    
    # Find the 15 nearest neighbors
    distances, indices = knn.kneighbors(X_copy[["lat", "long"]])
    
    # Calculate the average of the specified feature for the closest 15 houses
    X_copy[f"{feature}_closest15"] = np.mean(X_copy.iloc[indices.flatten()][feature].values.reshape(-1, 15), axis=1)
    
    return X_copy

def fill_missing_zipcodes_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing zipcodes using the zipcode of the geographically nearest neighbor.
    Requires columns: 'lat', 'long', and 'zipcode'.
    """

    known = df[df["zipcode"].notna()].copy()
    missing = df[df["zipcode"].isna()].copy()

    if missing.empty:
        return df

    # Prepare coordinates
    known_coords = known[["lat", "long"]].to_numpy()
    missing_coords = missing[["lat", "long"]].to_numpy()

    # Fit nearest neighbor model on known ZIP code entries
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(known_coords)
    distances, indices = nn.kneighbors(missing_coords)

    # Assign ZIP codes from nearest neighbors
    nearest_zips = known["zipcode"].to_numpy()[indices[:, 0]]
    df.loc[missing.index, "zipcode"] = nearest_zips
    return df


def remove_outliers(X_clean: pd.DataFrame, y_clean: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    # remove outliers
    threshold = 6e+6 # arbitrary threshold decided by looking at the data
    y_outliers_indexs = y_clean[(y_clean > threshold)].index
    # remove the outliers from the data
    X_clean = X_clean.drop(index=y_outliers_indexs).copy()
    y_clean = y_clean.drop(index=y_outliers_indexs).copy()
    return X_clean, y_clean







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

    This function defines and processes the command-line arguments required for the script.
    It ensures that if debug mode is enabled, an input file must be provided.

    Returns
    -------
    dict
        Dictionary containing the parsed arguments:
        - 'debug': bool, whether debug mode is active
        - 'input': str or None, input file path (required in debug mode)
        - 'feature_evaluation': bool, whether to enable feature evaluation
        - 'zipcode': bool, whether to add zipcode to features
        - 'first_part_only': bool, whether to run only the first part of the assignment
        - 'first_part': bool, whether to include the first part of the assignment
    """
    parser = argparse.ArgumentParser(description="House Price Prediction CLI")

    parser.add_argument( "-d", "--debug", action="store_true", help="Enable debug mode (requires --input)")
    parser.add_argument( "-z", "--zipcode", action="store_true", help="add zipcode to features")
    parser.add_argument( "-w", "--without_first_part", action="store_true", help="Run without the first part of the assignment")
    parser.add_argument( "-f", "--first_part_only", action="store_true", help="Run only the first part of the assignment")
    parser.add_argument( "-nf", "--no_feature_evaluation", action="store_true", help="Disable feature evaluation")
    parser.add_argument( "-i", "--input", type=str, help="Path to input CSV file (required in debug mode)")
    
    parser.add_argument( "-t", "--title", type=str, help="Title of the avarge loss plot (optional)")

    args = parser.parse_args()

    # Enforce: if debug is active, input is required
    if args.debug and not args.input:
        parser.error("--debug requires --input to be specified")

    features_evaluation = not args.no_feature_evaluation
    return {
        "debug": args.debug,
        "input": args.input, 
        "feature_evaluation": features_evaluation,
        "zipcode": args.zipcode,
        "first_part_only": args.first_part_only,
        "first_part": not args.without_first_part,
        "title": args.title


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
# Step 1: Identify duplicate houses by index (e.g., id)
    duplicate_indices = X_clean.index[X_clean.index.duplicated(keep=False)]

    if not duplicate_indices.empty:
        # Step 2: Extract only the duplicated rows
        duplicates_X = X_clean.loc[duplicate_indices]
        duplicates_y = y_clean.loc[duplicate_indices]

        # Step 3: Group by index — keep first row of features, average the targets
        grouped_X = duplicates_X.groupby(duplicates_X.index).first()
        grouped_y = duplicates_y.groupby(duplicates_y.index).mean()

        

        # Step 4: Remove all duplicate rows from original data
        X_clean = X_clean.drop(index=duplicate_indices)
        y_clean = y_clean.drop(index=duplicate_indices)


        # Step 5: Add back the grouped/cleaned duplicates
        X_clean = pd.concat([X_clean, grouped_X])
        y_clean = pd.concat([y_clean, grouped_y])

        # Optional: sort to maintain order by index
        X_clean = X_clean.sort_index()
        y_clean = y_clean.sort_index()


    return X_clean, y_clean

def convert_date_to_months(X_clean: pd.DataFrame) -> pd.DataFrame:
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

    X_clean["date"] = (
        ((X_clean["date"] - X_clean["date"].min()) / np.timedelta64(1, 'm'))
    )

    # Fill NaNs with the median
    median_months = X_clean["date"].median()
    X_clean["date"] = X_clean["date"].fillna(median_months).astype(int)

    return X_clean


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
    mask = (
    (X_clean["bedrooms"] > 0) &
    (X_clean["bathrooms"] > 0) &
    X_clean.notna().all(axis=1) &
    y_clean.notna()
    )

    X_clean = X_clean.loc[mask].copy()
    y_clean = y_clean.loc[mask].copy()
    

    

    # Step 2: Calculate avg room area
    avg_room_area = (
        X_clean["sqft_living"]
        - (X_clean["bathrooms"] * bathroom_area)
        - base_area
    ) / X_clean["bedrooms"]
    

    
    # Step 3: Remove rows with small bedrooms
    small_living = avg_room_area < min_bedroom_area
    mask = ~small_living  # Keep only rows with avg_room_area >= min_bedroom_area

    # Apply the mask consistently
    X_clean = X_clean.loc[mask].copy()
    y_clean = y_clean.loc[mask].copy()

    

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


    # Remove rows with NaN index and rows with NaN price
    mask = X_clean.notna().all(axis=1) & y.notna()
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]


    
    # make sure that the sqft_lot15 is positive - one value seems to be negative, probably a typo
    X_clean.loc[X_clean["sqft_lot15"] < 0, "sqft_lot15"] = (
        X_clean.loc[X_clean["sqft_lot15"] < 0, "sqft_lot15"].abs()
    )

    
    # Convert the 'date' column to a new feature representing months since the first sale -- need to be done in preprocess test also --
    
    update_price_center(X_clean, y_clean)
    X_clean = create_l2_dist_from_price_center(X_clean)

    # Set yr_renovated to 0 for houses where yr_built is later than yr_renovated
    X_clean.loc[X_clean["yr_built"] > X_clean["yr_renovated"], "yr_renovated"] = 0
    X_clean = convert_date_to_months(X_clean)
    

    
    
    # remove houses with unrealistically small living room size 
    X_clean, y_clean = remove_small_living_area(X_clean, y_clean)

    # Handle houses sold twice by taking the average of their features and target
    X_clean, y_clean = avarge_house_prices(X_clean, y_clean)

    X_clean = create_closest15_feature(X_clean, "grade")
    X_clean = create_closest15_feature(X_clean, "view")
    
    
    #check if date in the data

    


    X_clean, y_clean = remove_outliers(X_clean, y_clean)

    

    X = X.sort_index()
    y = y.sort_index()
    
    
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


    # Set yr_renovated to 0 for houses where yr_built is later than yr_renovated
    X_test.loc[X_test["yr_built"] > X_test["yr_renovated"], "yr_renovated"] = 0
    X_test = convert_date_to_months(X_test)
    X_test = create_closest15_feature(X_test, "grade")
    X_test = create_closest15_feature(X_test, "view")
    X_test = create_l2_dist_from_price_center(X_test)


    X_test.sort_index(inplace=True)
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




def split_data(X: pd.DataFrame, y: pd.Series, p: int = 75) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

    p : int
        Percentage of training data (default 75)

    Returns
    -------
    Tuple of:
        - X_train : pd.DataFrame
        - X_test  : pd.DataFrame
        - y_train : pd.Series
        - y_test  : pd.Series
    """
    seed = 42 # for reproducibility

    # Identify rows with NaN in index or y, or negative sqft_lot15
    mask = X.index.isna() | y.isna() | (X["sqft_lot15"] < 0)

    X_nan = X[mask]
    y_nan = y[mask]

    X_valid = X[~mask]
    y_valid = y[~mask]

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(X_valid.shape[0])
    split_point = int((p / 100) * len(shuffled_indices))

    train_idx = shuffled_indices[:split_point]
    test_idx = shuffled_indices[split_point:]

    X_train = pd.concat([X_valid.iloc[train_idx], X_nan])
    y_train = pd.concat([y_valid.iloc[train_idx], y_nan])

    X_test = X_valid.iloc[test_idx]
    y_test = y_valid.iloc[test_idx]

    return X_train, X_test, y_train, y_test



def Q_2_to_4(X: pd.DataFrame, y: pd.Series, args: dict) -> NoReturn:
    """
    Implement the Q2 to Q4 of the assignment.
    1. Split the data into train and test sets (75% train, 25% test).
       - add zipcode column based on lat/long according to the command line argument. my addition to the data. done only by command due to long running time.
    2. Preprocess the training data.
    3. evaluate the features using the feature_evaluation function.
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix of regression problem
    y : pd.Series
        Response vector to split
    args : dict
        Dictionary containing command-line arguments.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, p=75)
    if args["zipcode"]:
        print("Adding zipcode column based on lat/long...")
        X_train["zipcode"] = X_train.apply(lambda x: get_zipcode(x["lat"], x["long"]), axis=1)
        X_test["zipcode"] = X_test.apply(lambda x: get_zipcode(x["lat"], x["long"]), axis=1)


    # check if zipcode is a column in the data
    if "zipcode" in X_train.columns and "zipcode" in X_test.columns:
        # Fill missing zipcodes using the zipcode of the geographically nearest neighbor - in the X_train
        X_train["zipcode"] = pd.to_numeric(X_train["zipcode"], errors="coerce").astype("Int64")
        X_train = fill_missing_zipcodes_by_proximity(X_train)

        # Fill missing zipcodes with 0 in the X_test
        X_test["zipcode"] = pd.to_numeric(X_test["zipcode"], errors="coerce").astype("Int64")
        X_test["zipcode"] = X_test["zipcode"].fillna(0)

        

    X = X.sort_index()
    y = y.sort_index()
    
    X_train, y_train = preprocess_train(X_train, y_train)

    if args["feature_evaluation"]:
        feature_evaluation(X_train, y_train, output_path="feature_evaluation")


def sample_train_set(X_train: pd.DataFrame, y_train: pd.Series, p: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Sample a subset of the training set based on the given percentage.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training design matrix.
        y_train : pd.Series
            Training response vector.
        p : int
            Percentage of the training data to sample.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Sampled training design matrix and response vector.
        """
        seed = os.getpid()  # Use the process ID for the seed
        np.random.seed(seed)  # Ensure reproducibility
        sample_size = int((p / 100) * len(X_train))
        sampled_indices = np.random.choice(X_train.index, size=sample_size, replace=False)

        X_sampled = X_train.loc[sampled_indices]
        y_sampled = y_train.loc[sampled_indices]

        return X_sampled, y_sampled

def run_single_experiment( args: dict,  X_train: pd.DataFrame,  y_train: pd.Series,  X_test: pd.DataFrame,  y_test: pd.Series,  p: int,  i: int
                ) -> Tuple[float, float]:
    """
    Run a single experiment for training and evaluating a linear regression model.

    Parameters
    ----------
    args : dict
        Dictionary containing command-line arguments.
    X_train : pd.DataFrame
        Training design matrix.
    y_train : pd.Series
        Training response vector.
    X_test : pd.DataFrame
        Test design matrix.
    y_test : pd.Series
        Test response vector.
    p : int
        Percentage of training data to use.
    i : int
        Experiment iteration index.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - Loss (float): The squared error loss on the test set.
        - Variance (float): The variance of predictions on the test set.
    """
    model = LinearRegression(include_intercept=True)

    X_train, y_train = sample_train_set(X_train, y_train, p=p)

    if args["zipcode"]:
        X_train["zipcode"] = X_train.apply(lambda x: get_zipcode(x["lat"], x["long"]), axis=1)
        X_test["zipcode"] = X_test.apply(lambda x: get_zipcode(x["lat"], x["long"]), axis=1)

    # Check if zipcode is a column in the data
    if "zipcode" in X_train.columns and "zipcode" in X_test.columns:
        # Fill missing zipcodes using the zipcode of the geographically nearest neighbor - in the X_train
        X_train["zipcode"] = pd.to_numeric(X_train["zipcode"], errors="coerce").astype("Int64")
        X_train = fill_missing_zipcodes_by_proximity(X_train)

        # Fill missing zipcodes with 0 in the X_test
        X_test["zipcode"] = pd.to_numeric(X_test["zipcode"], errors="coerce").astype("Int64")
        X_test["zipcode"] = X_test["zipcode"].fillna(0)
    
    X_train, y_train = preprocess_train(X_train, y_train)
    X_test = preprocess_test(X_test)

    X_diff = X_test[X_test.columns.difference(X_train.columns)]
    if not X_diff.empty:
        raise ValueError(f"Test data has extra columns: {X_diff.columns.tolist()}")

    model.fit(X_train.to_numpy(), np.squeeze(y_train.to_numpy()))
    return model.loss(X_test.to_numpy(), np.squeeze(y_test.to_numpy())), model.var(X_test.to_numpy(), np.squeeze(y_test.to_numpy()))



def main() -> NoReturn:
    """
    Main function to run the house price prediction pipeline.

    Parameters
    ----------
    args : dict
        Dictionary containing command-line arguments.
    """
    # Load data
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


    # implement the Q2 to Q4
    if args["first_part"]:    
        Q_2_to_4(X, y, args)
    if args["first_part_only"]:
        print("First part only, exiting...")
        return
    



    # creating the linear regression model

    loss, var = np.zeros(100 - 10 + 1), np.zeros(100 - 10 + 1)
    std_loss, std_var = np.zeros(100 - 10 + 1), np.zeros(100 - 10 + 1)

    # Define percentages for training data sizes
    percentages = np.arange(10, 101)

    X_train, X_test, y_train, y_test = split_data(X, y, p=75)
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    for p in percentages:
        print(f"\rRunning experiments for {p}% of the data...", end="")

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(
                run_single_experiment,
                [(args, X_train, y_train, X_test, y_test, p, i) for i in range(10)]
            )

        cur_loss, cur_var = zip(*results)  # unzip the results
        cur_loss, cur_var = np.array(cur_loss), np.array(cur_var)

        # Store average and variance of loss over test set
        # check if some of the arrays have NaN values or have empty values
        loss[p - 10], var[p - 10], std_loss[p - 10], std_var[p - 10] = cur_loss.mean(), cur_var.mean(), cur_loss.std(), cur_var.std()


    # Create the plot
    fig = go.Figure()

    # Add the average loss line
    fig.add_trace(go.Scatter(x=percentages, y=loss, mode='lines', name='Average Loss', line=dict(color='blue')))

    # Add the error ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([percentages, percentages[::-1]]),
        y=np.concatenate([loss - 2 * std_loss, (loss + 2 * std_loss)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),  # Ensure the ribbon has no border
        hoverinfo="skip",
        name='Error Ribbon (±2*std)'
    ))

    title = "Average Loss" if not args["title"] else args["title"]
    # Update layout
    fig.update_layout(
        title=title + " as a Function of Training Size",
        xaxis_title="Percentage of Training Data",
        yaxis_title="Loss (Squared Error)",  # Add units to the y-axis
        legend=dict(x=0, y=1),
        template="plotly_white",
        yaxis=dict(
            type="log",  # Use logarithmic scale
            title="Loss (e^scale)",  # Update y-axis title
            exponentformat="e"  # Format exponents as e^x
        )
    )

    # Save the plot as an image
    
    fig.write_image(title + ".png")

    # Show the plot
    fig.show()

    return



if __name__ == '__main__':
    main()
