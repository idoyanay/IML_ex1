import plotly.express as px
import numpy as np
import pandas as pd
import argparse
from typing import NoReturn , Tuple
from polynomial_fitting import PolynomialFitting

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    Temp = pd.read_csv(filename, parse_dates=["Date"])

    Temp["DayOfYear"] = Temp["Date"].dt.dayofyear

    # Ensure required columns exist
    required_columns = ["Day", "Month", "Temp"]
    for col in required_columns:
        if col not in Temp.columns:
            raise ValueError(f"Missing required column: {col}")

    # remove temp outliers
    temp_threshold = 50
    Temp = Temp[abs(Temp["Temp"]) < temp_threshold]
    

    # mask for invalid dates
    invalid_date_mask = (
        (Temp["DayOfYear"] > 366) |
        (Temp["DayOfYear"] < 1) |
        (Temp["Day"] > 31) |
        (Temp["Day"] < 1) |
        (Temp["Month"] > 12) |
        (Temp["Month"] < 1)
    )

    Temp = Temp[~invalid_date_mask]

    
    # remove rows with NaNs
    Temp = Temp.dropna()
    # remove rows with Infs
    Temp = Temp[Temp["Temp"].apply(lambda x: x != float("inf"))]

    

    return Temp



def explore_country(country: str, Temp: pd.DataFrame) -> pd.DataFrame:
    """
    Explore data for a specific country.
    Parameters
    ----------
    country: str
        Country to explore

    Temp: pd.DataFrame
        Dataframe containing the data

    Returns
    -------
    Filtered dataframe for the specified country
    """
    

    # --- Filter Israel-only data ---
    israel_Temp = Temp[Temp["Country"] == "Israel"].copy()

    # --- Plot 1: Scatter plot of Temp vs DayOfYear colored by Year ---
    fig_scatter = px.scatter(
        israel_Temp,
        x="DayOfYear",
        y="Temp",
        color=israel_Temp["Year"].astype(str),
        title="Daily Temperature in Israel by Day of Year",
        labels={"Temp": "Temperature (°C)", "DayOfYear": "Day of Year", "color": "Year"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_scatter.write_image("israel_temp_by_dayofyear_scatter.png")  # Save as static image
    fig_scatter.show()  # Optional: open in browser
    # save plot as png
    fig_scatter.write_image(f"{country}_temp_by_dayofyear_scatter.png")


    # --- Plot 2: Bar plot of std(Temp) by Month ---


    monthly_std = israel_Temp.groupby("Month")["Temp"].agg("std").reset_index(name="TempStd")

    

    fig_bar = px.bar(
        monthly_std,
        x="Month",
        y="TempStd",
        title="Monthly Temperature Standard Deviation in Israel",
        labels={"Month": "Month", "TempStd": "Temperature Std. Dev. (°C)"},
        color_discrete_sequence=px.colors.qualitative.Set1  # Use a predefined color sequence
    )
    fig_bar.write_image("israel_temp_monthly_std_barplot.png")
    fig_bar.show()

    # save plot as png
    fig_bar.write_image(f"{country}_temp_monthly_std_barplot.png")



    return israel_Temp


def explore_all_countries(Temp: pd.DataFrame) -> NoReturn:
    """
    Explore differences between countries.
    Parameters
    ----------
    Temp: pd.DataFrame
        Dataframe containing the data

    Returns
    -------
    NoReturn
    """

    # Group by Country and Month, calculate mean and std of Temp
    monthly_stats = Temp.groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()
    monthly_stats.rename(columns={"mean": "TempMean", "std": "TempStd"}, inplace=True)


    # --- Plot 1: Plot using plotly.express.line with error bars
    fig = px.line(
        monthly_stats,
        x="Month",
        y="TempMean",
        color="Country",
        error_y="TempStd",
        title="Average Monthly Temperature with Std Deviation (by Country)",
        labels={"TempMean": "Average Temperature (°C)", "Month": "Month"}
    )

    fig.write_image("monthly_temp_with_error_by_country.png")
    fig.show()

    # --- Plot 2: Heatmap of TempMean by Country and Month ---
    pivot_mean = monthly_stats.pivot(index="Country", columns="Month", values="TempMean")
    fig_mean = px.imshow(
        pivot_mean,
        labels=dict(color="Mean Temp (°C)", x="Month", y="Country"),
        title="Heatmap of Mean Monthly Temperature by Country"
    )
    fig_mean.write_image("heatmap_mean_temperature.png")
    fig_mean.show()

    # --- Plot 3: Heatmap of TempStd by Country and Month ---
    pivot_std = monthly_stats.pivot(index="Country", columns="Month", values="TempStd")
    fig_std = px.imshow(
        pivot_std,
        labels=dict(color="Temp Std Dev (°C)", x="Month", y="Country"),
        title="Heatmap of Monthly Temperature Std devided by Country"
    )
    fig_std.write_image("heatmap_std_temperature.png")
    fig_std.show()

    return

def split_data(Temp: pd.DataFrame, p: float, seed:int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Split the data into train and test sets
    X, y = Temp.drop(columns=["Temp"]), Temp["Temp"]
    # leave only the DayOfYear column
    X = X[["DayOfYear"]]



    
    # Shuffle and split
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(X.shape[0])
    split_point = int((p / 100) * len(shuffled_indices))

    # Split the data
    train_idx = shuffled_indices[:split_point]
    test_idx = shuffled_indices[split_point:]

    # Create train and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test

def explore_different_k(Temp: pd.DataFrame) -> NoReturn:
    """
    Explore differences between countries.
    Parameters
    ----------
    Temp: pd.DataFrame
        Dataframe containing the data

    Returns
    -------
    NoReturn
    """

    p = 75 # the percentile to use for the train model
    # Split the data into train and test sets - X is only the DayOfYear column
    X_train, X_test, y_train, y_test= split_data(Temp, p)
    upper_bound = 10
    loss_k = np.zeros(upper_bound)
    for k in range(1, upper_bound + 1):
        # Fit the model


        model = PolynomialFitting(k=k)
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        # Evaluate the model

        loss_k[k-1] = round(model.loss(X_test.to_numpy(), y_test.to_numpy()), 2)

    fig = px.bar(
        x=np.arange(1, upper_bound + 1),
        y=loss_k,
        title="Loss for different k values",
        labels={"x": "k", "y": "Loss"},
    )

    fig.update_layout(
        xaxis_title="k",
        yaxis_title="Loss",
        title_x=0.5,
        title_y=0.95
    )

    fig.show()
    fig.write_image("loss_for_different_k.png")

    
    # print the loss for each k value
    for k, loss in enumerate(loss_k, start=1):
        print(f"k={k}: Loss={loss}")

    return

def parse_arguments() -> dict:
    """
    Parses command-line arguments for the house price prediction script.

    If debug mode is enabled, an input file must be provided.

    Returns
    -------
    dict
        Dictionary containing parsed arguments:
        - "all": bool, whether to run all questions
        - "question_3": bool, whether to run question 3
        - "question_4": bool, whether to run question 4
        - "question_5": bool, whether to run question 5
        - "question_6": bool, whether to run question 6
    """
    parser = argparse.ArgumentParser(description="City Temperature Prediction CLI")

    parser.add_argument(
        "-q", "--questions", type=str, help="Specify the question(s) to run. Use comma-separated values (e.g., '3,4') or ranges (e.g., '3-5')."
    )

    args = parser.parse_args()

    # Parse the questions argument
    selected_questions = set()
    if args.questions:
        for part in args.questions.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                selected_questions.update(range(start, end + 1))
            else:
                selected_questions.add(int(part))

    # Fill the dictionary based on selected questions
    not_all = len(selected_questions) == 0
    return {
        "question_3": 3 in selected_questions or not_all,
        "question_4": 4 in selected_questions or not_all,
        "question_5": 5 in selected_questions or not_all,
        "question_6": 6 in selected_questions or not_all
    }


def check_israel_model_on_others(Temp:  pd.DataFrame, k: int) -> NoReturn:
    """
        Evaluates the performance of a polynomial regression model trained on Israeli data
        when applied to data from other countries. The function calculates the loss for
        each country and visualizes the results in a bar plot.
        Args:
            Temp (pd.DataFrame): A DataFrame containing temperature data with at least the
                                 following columns: "Country", "DayOfYear", and the target variable.
            k (int): The degree of the polynomial to be used in the PolynomialFitting model.
        Returns:
            NoReturn: This function does not return a value. It generates and displays a plot
                      of the model's loss for each country (excluding Israel) and saves it as
                      a PNG file.
    """

    p = 100 # the percentile to use for the train model
    # Split the data into train and test sets - X is only the DayOfYear column
    X_train, _, y_train, _= split_data(Temp, p)
    # Fit the model
    model = PolynomialFitting(k=k)
    model.fit(X_train.to_numpy(), y_train.to_numpy().flatten())
    # Evaluate the model
    countries_loss = []

    for country in Temp["Country"].unique():
        if country == "Israel":
            continue
        # Filter the data for the current country
        country_data = Temp[Temp["Country"] == country].copy()
        # Split the data into train and test sets - X is only the DayOfYear column
        _, X_test, _, y_test = split_data(country_data, 0)
        # Evaluate the model
        loss = round(model.loss(X_test.to_numpy(), y_test.to_numpy().flatten()), 2)
        countries_loss.append((loss, country))
        print(f"Loss for {country}: {loss}")

    # Convert to DataFrame for easier plotting
    loss_df = pd.DataFrame(countries_loss, columns=["Loss", "Country"])

    # Plot a bar plot of the loss for each country
    fig = px.bar(
        loss_df,
        x="Country",
        y="Loss",
        title="Model Loss for Each Country (Excluding Israel)",
        labels={"Loss": "Loss", "Country": "Country"},
        color="Loss",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.write_image("loss_by_country.png")
    fig.show()
    

if __name__ == '__main__':

    args = parse_arguments()
    print(args)

    # Question 2 - Load and preprocessing of city temperature dataset
    Temp = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    if args["question_3"]:
        print("runnning question 3")
        Temp_israel = explore_country("israel", Temp)

    # Question 4 - Exploring differences between countries
    if args["question_4"]:
        print("running question 4")
        explore_all_countries(Temp)

    # Question 5 - Fitting model for different values of `k`
    if args["question_5"]:
        print("running question 5")
        Temp_israel = Temp[Temp["Country"] == "Israel"].copy()
        explore_different_k(Temp_israel)

    # Question 6 - Evaluating fitted model on different countries
    if args["question_6"]:
        print("running question 6")
        # Split the data into train and test sets - X is only the DayOfYear column
        best_k = 5 # best k founded in question 5
        check_israel_model_on_others(Temp, best_k)

    pass
