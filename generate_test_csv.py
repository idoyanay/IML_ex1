import pandas as pd
import argparse

def create_test_csv(input_path: str, output_path: str, n_rows: int = 500, random: bool = True, seed: int = 42) -> None:
    """
    Creates a smaller CSV file from a larger one for quick testing.

    Parameters
    ----------
    input_path : str
        Path to the original large CSV file

    output_path : str
        Path where the smaller test CSV will be saved

    n_rows : int
        Number of rows to include in the test CSV

    random : bool
        Whether to randomly sample the rows (default True)

    seed : int
        Random seed for reproducibility
    """
    df = pd.read_csv(input_path)

    if random:
        df_sample = df.sample(n=n_rows, random_state=seed)
    else:
        df_sample = df.head(n_rows)

    df_sample.to_csv(output_path, index=False)
    print(f"✅ Test CSV with {len(df_sample)} rows saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a small test CSV from a larger dataset.")
    parser.add_argument("input", help="Path to the original CSV file")
    parser.add_argument("output", help="Path to save the smaller test CSV")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to include in the test file (default: 500)")
    parser.add_argument("--no-random", action="store_true", help="Use the first N rows instead of random sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    create_test_csv(
        input_path=args.input,
        output_path=args.output,
        n_rows=args.rows,
        random=not args.no_random,
        seed=args.seed
    )
