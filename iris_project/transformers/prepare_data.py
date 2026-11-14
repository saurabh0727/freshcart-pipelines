import pandas as pd
from sklearn.model_selection import train_test_split


def transform(df, test_size=0.2, target_col="target", random_state=42):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    result = df.copy()

    if df.empty:
        result["split"] = pd.Series(dtype="object")
        return result

    target = df[target_col] if target_col in df.columns else None
    stratify = None

    if target is not None and len(df) > 1 and target.nunique() > 1:
        stratify = target

    _, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    result["split"] = "train"
    result.loc[test_idx, "split"] = "test"

    return result