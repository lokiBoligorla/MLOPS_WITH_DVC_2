
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def stage_02_preprocessing_eda():

    print("\nSTAGE 2: DATA PREPROCESSING & EDA")
    print("=" * 50)

    # Step 1: Load raw dataset
    input_path = "data/raw/dataset.csv"

    df = pd.read_csv(input_path)

    print("Original Dataset Shape:", df.shape)

    # Step 2: Handle infinite values
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Step 3: Remove duplicate records
    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    print("Duplicates Removed:", duplicate_count)

    # Step 4: Display missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Step 5: Save cleaned dataset
    os.makedirs("data/processed", exist_ok=True)

    cleaned_path = "data/processed/cleaned_data.csv"

    df.to_csv(cleaned_path, index=False)

    print("\nCleaned Dataset Saved:", cleaned_path)

    # Step 6: Create EDA directory
    eda_dir = "artifacts/eda"

    os.makedirs(eda_dir, exist_ok=True)

    # Step 7: Descriptive statistics
    statistics = df.describe()

    statistics.to_csv(
        os.path.join(eda_dir, "descriptive_statistics.csv")
    )

    # Step 8: Correlation analysis
    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    correlation.to_csv(
        os.path.join(eda_dir, "correlation_matrix.csv")
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(eda_dir, "correlation_heatmap.png")
    )

    plt.close()

    # Step 9: Histograms and boxplots
    for column in numeric_df.columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True
        )

        plt.title(f"{column} Distribution")
        plt.tight_layout()

        plt.savefig(
            os.path.join(eda_dir, f"{column}_distribution.png")
        )

        plt.close()

        plt.figure(figsize=(8, 5))

        sns.boxplot(data=df, x=column)

        plt.title(f"{column} Boxplot")
        plt.tight_layout()

        plt.savefig(
            os.path.join(eda_dir, f"{column}_boxplot.png")
        )

        plt.close()

    # Step 10: Feature vs target analysis
    target_column = "MedHouseVal"

    for column in numeric_df.columns:

        if column == target_column:
            continue

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=df,
            x=column,
            y="median_house_value",
            alpha=0.4
        )

        plt.title(f"{column} vs {target_column}")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                eda_dir,
                f"{column}_vs_{target_column}.png"
            )
        )

        plt.close()

    print("\nEDA Reports Generated Successfully!")

    print("EDA Output Directory:", eda_dir)

    print("\nFinal Dataset Shape:", df.shape)

    print("\nStage 2 Completed Successfully!")

    return cleaned_path


if __name__ == "__main__":
    stage_02_preprocessing_eda()
