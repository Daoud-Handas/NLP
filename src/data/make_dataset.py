import pandas as pd

def make_dataset(filename):
    # create two dataframe for train and test 80% and 20%
    return pd.read_csv(filename)

def make_train_test_split(filename):
    # create two dataframe for train and test 80% and 20%
    df = pd.read_csv(filename)
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)
    # save it csv
    df_train.to_csv("data/raw/train.csv", index=False)
    df_test.to_csv("data/raw/test.csv", index=False)