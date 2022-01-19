import pandas as pd

def clean_data(df):
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    return df


if __name__ == '__main__':
    raw_df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    clean_df = clean_data(raw_df)
    clean_df.to_csv("data/clean/clean_census.csv", index=False)