
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def stage_03_feature_engineering_splitting():

    print("\nSTAGE 3: FEATURE ENGINEERING & DATA SPLITTING")
    print("=" * 50)

    # Step 1: Load cleaned dataset
    input_path = "data/processed/cleaned_data.csv"

    df = pd.read_csv(input_path)

    print("Original Dataset Shape:", df.shape)

    # Step 2: Create new features
    print("\nStarting Feature Engineering...")

    if {"AveBedrms", "AveRooms"}.issubset(df.columns):

        df["Bedroom_Ratio"] = (
            df["AveBedrms"] /
            df["AveRooms"].replace(0, np.nan)
        )

    if {"AveRooms", "AveOccup"}.issubset(df.columns):

        df["Rooms_Per_Occupant"] = (
            df["AveRooms"] /
            df["AveOccup"].replace(0, np.nan)
        )

    # Step 3: Apply log transformation
    log_columns = [
        "Population",
        "AveRooms",
        "AveOccup"
    ]

    for column in log_columns:

        if column in df.columns:

            if (df[column].dropna() < 0).any():
                raise ValueError(
                    f"Cannot apply log1p to negative values in {column}"
                )

            df[f"{column}_Log"] = np.log1p(df[column])

    print("Feature Engineering Completed!")

    print("\nNew Dataset Shape:", df.shape)

    # Step 4: Save engineered dataset
    os.makedirs("data/processed", exist_ok=True)

    engineered_path = (
        "data/processed/engineered_data.csv"
    )

    df.to_csv(engineered_path, index=False)

    print("Engineered Dataset Saved:", engineered_path)

    # Step 5: Separate features and target
    target_column = "median_house_value"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("\nFeatures Shape:", X.shape)
    print("Target Shape:", y.shape)

    # Step 6: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Step 7: Save split datasets
    X_train.to_csv(
        "data/processed/X_train.csv",
        index=False
    )

    X_test.to_csv(
        "data/processed/X_test.csv",
        index=False
    )

    y_train.to_frame(name=target_column).to_csv(
        "data/processed/y_train.csv",
        index=False
    )

    y_test.to_frame(name=target_column).to_csv(
        "data/processed/y_test.csv",
        index=False
    )

    print("\nData Splitting Completed!")

    print("Training Features:", X_train.shape)
    print("Testing Features:", X_test.shape)

    print("Training Target:", y_train.shape)
    print("Testing Target:", y_test.shape)

    print("\nAll Files Saved Successfully!")

    print("\nStage 3 Completed Successfully!")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    stage_03_feature_engineering_splitting()
