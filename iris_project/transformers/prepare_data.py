import pandas as pd
from sklearn.model_selection import train_test_split


def transform(df, test_size=0.2):

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=42,
        stratify=df["target"]
    )

    result = df.copy()

    result["split"] = "train"
    result.loc[test_idx, "split"] = "test"

    return result