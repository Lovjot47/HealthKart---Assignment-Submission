import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nSample:\n", df.head())
    return df

if __name__ == "__main__":
    df = load_data("data/GrammarandProductReviews.csv")
