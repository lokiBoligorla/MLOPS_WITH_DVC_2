
import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor


def stage_04_model_training():

    print("\nSTAGE 4: MODEL TRAINING & FEATURE SELECTION")
    print("=" * 55)

    # Step 1: Load training data
    X_train = pd.read_csv("data/processed/X_train.csv")

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    )["median_house_value"]

    print("Training Features:", X_train.shape)
    print("Training Target:", y_train.shape)

    # Step 2: Identify numerical and categorical columns
    numerical_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print("\nNumerical Columns:", numerical_columns)
    print("Categorical Columns:", categorical_columns)

    # Step 3: Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Step 4: Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Step 5: Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    # Step 6: Complete ML pipeline
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),

        ("feature_selection", SelectKBest(
            score_func=f_regression,
            k=12
        )),

        ("model", RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Step 7: Train model
    print("\nTraining Random Forest Regressor...")

    model_pipeline.fit(X_train, y_train)

    print("Model Training Completed!")

    # Step 8: Get selected feature names
    preprocessed_feature_names = (
        model_pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    selector = model_pipeline.named_steps["feature_selection"]

    selected_features = preprocessed_feature_names[
        selector.get_support()
    ].tolist()

    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    # Step 9: Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Step 10: Save complete pipeline
    model_path = "model.pkl"

    joblib.dump(model_pipeline, model_path)

    # Step 11: Save selected feature names
    with open(
        "artifacts/selected_features.json", "w"
    ) as file:
        json.dump(selected_features, file, indent=4)

    print("\nModel Saved:", model_path)
    print("Selected Features Saved Successfully!")

    print("\nStage 4 Completed Successfully!")

    return model_path


if __name__ == "__main__":
    stage_04_model_training()
