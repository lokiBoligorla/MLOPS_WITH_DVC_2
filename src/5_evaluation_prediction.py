
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def stage_05_evaluation_prediction():

    print("\nSTAGE 5: MODEL EVALUATION & PREDICTION")
    print("=" * 55)

    # Step 1: Load testing data
    X_test = pd.read_csv("data/processed/X_test.csv")

    y_test = pd.read_csv(
        "data/processed/y_test.csv"
    )["median_house_value"]

    print("Testing Features:", X_test.shape)
    print("Testing Target:", y_test.shape)

    # Step 2: Load trained pipeline
    model_path = "model.pkl"

    model = joblib.load(model_path)

    print("\nTrained Model Loaded Successfully!")

    # Step 3: Generate predictions
    print("\nGenerating Predictions...")

    predictions = model.predict(X_test)

    # Step 4: Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # Step 5: Display evaluation results
    print("\nModel Evaluation Results")
    print("-" * 40)

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    # Step 6: Save metrics
    os.makedirs("artifacts", exist_ok=True)

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Score": float(r2)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Step 7: Save predictions
    predictions_df = pd.DataFrame({
        "Actual_Value": y_test.to_numpy(),
        "Predicted_Value": predictions
    })

    predictions_df.to_csv(
        "artifacts/predictions.csv",
        index=False
    )

    print("\nPredictions Saved Successfully!")

    # Step 8: Actual vs Predicted plot
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.4
    )

    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual House Values")
    plt.ylabel("Predicted House Values")
    plt.title("Actual vs Predicted Values")

    plt.tight_layout()

    plt.savefig("artifacts/actual_vs_predicted.png")
    plt.close()

    # Step 9: Residual plot
    residuals = y_test.to_numpy() - predictions

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.4
    )

    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted House Values")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")

    plt.tight_layout()

    plt.savefig("artifacts/residual_plot.png")
    plt.close()

    print("\nEvaluation Metrics Saved:", "artifacts/metrics.json")
    print("Prediction File:", "artifacts/predictions.csv")
    print("\nStage 5 Completed Successfully!")

    return metrics


if __name__ == "__main__":
    stage_05_evaluation_prediction()
