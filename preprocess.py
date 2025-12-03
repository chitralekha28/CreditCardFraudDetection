import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/creditcard.csv"):
    df = pd.read_csv(path)

    # Feature Engineering
    df["Hour"] = df["Time"] % 24

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("Class", axis=1))
    y = df["Class"].values

    return X, y, scaler
