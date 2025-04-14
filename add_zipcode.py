import pandas as pd
import argparse
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
from tqdm import tqdm

def get_zipcode(lat: float, lon: float) -> str:
    try:
        geolocator = Nominatim(user_agent="zipcode-batch-script")
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        return location.raw["address"].get("postcode", None) if location else None
    except GeopyError as e:
        print(f"Failed for ({lat}, {lon}): {e}")
        return None

def add_zipcodes(df: pd.DataFrame) -> pd.DataFrame:
    print("Looking up zipcodes...")

    tqdm.pandas(desc="Resolving zipcodes")
    df["zipcode"] = df.progress_apply(lambda row: get_zipcode(row["lat"], row["long"]), axis=1)

    # Convert to numeric zipcode (nullable integer)
    df["zipcode"] = pd.to_numeric(df["zipcode"], errors="coerce").astype("Int64")

    return df

def main():
    parser = argparse.ArgumentParser(description="Add zipcode column to CSV file based on lat/long.")
    parser.add_argument("input_csv", help="Path to the input CSV file (must have 'lat' and 'long' columns)")
    parser.add_argument("output_csv", help="Path to the output CSV file")

    args = parser.parse_args()

    # Load input data
    df = pd.read_csv(args.input_csv)

    if "lat" not in df.columns or "long" not in df.columns:
        raise ValueError("Input file must contain 'lat' and 'long' columns.")

    # Add zipcodes
    df_with_zip = add_zipcodes(df)

    # Save to new CSV
    df_with_zip.to_csv(args.output_csv, index=False)

    if df_with_zip["zipcode"].isnull().any():
        raise ValueError(f"coudn't find zipcode for row: {df_with_zip[df_with_zip['zipcode'].isnull()]}")
    # Save the new CSV with zipcodes
    print(f"✅ Saved new CSV with zipcode column to: {args.output_csv}")

if __name__ == "__main__":
    main()
