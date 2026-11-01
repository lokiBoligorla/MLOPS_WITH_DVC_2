
import os
import json
import numpy as np
import pandas as pd


def stage_01_ingestion_validation():

    print("\nSTAGE 1: DATA INGESTION & VALIDATION")
    print("=" * 50)

    # Step 1: Dataset URL
    dataset_url = (
        "https://raw.githubusercontent.com/"
        "ageron/handson-ml2/master/datasets/housing/housing.csv"
    )

    # Step 2: Download dataset directly from URL
    print("\nDownloading California Housing Dataset...")

    df = pd.read_csv(dataset_url)

    print("Dataset Downloaded Successfully!")
    print("Dataset Shape:", df.shape)

    # Step 3: Create output directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Step 4: Save raw dataset
    raw_path = "data/raw/dataset.csv"

    df.to_csv(raw_path, index=False)

    print("\nRaw Dataset Saved:", raw_path)

    # Step 5: Data validation
    print("\nStarting Data Validation...")

    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()

    numeric_df = df.select_dtypes(include=np.number)

    infinite_values = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    numerical_columns = numeric_df.columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Step 6: Generate validation report
    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {
            col: int(value)
            for col, value in missing_values.items()
        },
        "duplicate_records": int(duplicate_count),
        "infinite_values": infinite_values,
        "data_types": data_types,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns
    }

    # Step 7: Save validation report
    report_path = "artifacts/validation_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nValidation Results")
    print("-" * 40)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Missing Values:", int(missing_values.sum()))
    print("Duplicate Records:", duplicate_count)
    print("Infinite Values:", infinite_values)

    print("\nValidation Report Saved:", report_path)

    print("\nStage 1 Completed Successfully!")

    return raw_path


if __name__ == "__main__":
    stage_01_ingestion_validation()
