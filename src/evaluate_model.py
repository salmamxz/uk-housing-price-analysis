# This file handles the evaluation stage of our machine learning project.
#
# Step 7: Use the trained model to make predictions.
# Step 8: Evaluate predictions using MAE, RMSE and R².
# Step 9: Analyse residuals to understand where the model makes errors.
#
# We also generate and save visualisations to help us
# understand how well our model performs.


from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_pipeline():

    # Find the main project folder and define where
    # our saved model, test dataset and figures are located.
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "baseline_linear_model.joblib"
    test_path = project_root / "data" / "processed" / "test_set_eval.csv"

    figures_dir = project_root / "reports" / "figures"

    # Create the figures folder if it doesn't already exist.
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Before continuing, check that both required files exist.
    # If they are missing, we need to run train_model.py first.
    if not model_path.exists() or not test_path.exists():
        print("[ERROR] Missing model or test dataset. Run src/train_model.py first.")
        return

    # Load our previously trained model and the saved test dataset.
    # This means we can evaluate the model without training it again.
    print("[INFO] Loading model and test set...")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    # Separate the actual prices from the property features.
    # We need both so we can compare actual prices with predictions.
    y_test = test_df["actual_price"]
    X_test = test_df.drop(columns=["actual_price"])


    # ---------------------------------------------------------
    # STEP 7: MAKE PREDICTIONS
    # ---------------------------------------------------------

    # Pass the unseen property features into our trained pipeline.
    # The pipeline automatically applies the same preprocessing
    # before Linear Regression generates the predicted prices.
    print("[INFO] Generating predictions on test set...")

    y_pred = model.predict(X_test)


    # ---------------------------------------------------------
    # STEP 8: EVALUATE MODEL PERFORMANCE
    # ---------------------------------------------------------

    # Compare the predicted prices against the actual prices.
    #
    # MAE: Average absolute difference between predictions
    # and actual prices.
    #
    # RMSE: Similar to MAE, but penalises larger errors more.
    #
    # R²: Measures how much variation in the target prices
    # is explained by our model.
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    # Display all three metrics in a readable format.
    print("\n" + "=" * 45)
    print("         MODEL PERFORMANCE METRICS")
    print("=" * 45)

    print(f"Mean Absolute Error (MAE) : £{mae:,.2f}")
    print(f"Root Mean Squared Error   : £{rmse:,.2f}")
    print(f"R² Score                  : {r2:.4f}")

    print("=" * 45 + "\n")


    # ---------------------------------------------------------
    # STEP 9: RESIDUAL ANALYSIS
    # ---------------------------------------------------------

    # A residual is the difference between the actual
    # price and the price predicted by our model.
    #
    # Positive residual = model underestimated the price.
    # Negative residual = model overestimated the price.
    # Zero residual = prediction was exactly correct.
    residuals = y_test - y_pred

    # Set the visual style for our plots.
    sns.set_theme(style="whitegrid")

    # Create two plots next to each other.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    # ---------------------------------------------------------
    # PLOT 1: RESIDUALS VS PREDICTED PRICES
    # ---------------------------------------------------------

    # We have over 83,000 test records.
    # Plotting all of them can make the chart difficult to read.
    #
    # So we randomly select up to 5,000 points.
    # random_state=42 makes the sample reproducible.
    plot_idx = np.random.RandomState(42).choice(
        len(y_test),
        size=min(5000, len(y_test)),
        replace=False
    )

    # Create a scatter plot showing how prediction errors
    # change as predicted property prices increase.
    #
    # We divide by 1,000 to display prices in thousands of pounds.
    sns.scatterplot(
        x=y_pred[plot_idx] / 1_000,
        y=residuals.iloc[plot_idx] / 1_000,
        alpha=0.3,
        color="crimson",
        ax=axes[0],
    )

    # Add a horizontal line at zero.
    # Ideally, residuals should be randomly distributed around it.
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.2)

    axes[0].set_title("Residuals vs. Fitted Values (Sample)")
    axes[0].set_xlabel("Predicted Price (£'000)")
    axes[0].set_ylabel("Residual (£'000)")


    # ---------------------------------------------------------
    # PLOT 2: DISTRIBUTION OF RESIDUALS
    # ---------------------------------------------------------

    # This histogram shows how our prediction errors
    # are distributed across the test dataset.
    #
    # We want to see whether the errors are roughly
    # centred around zero and approximately normally distributed.
    #
    # The KDE line gives us a smooth view of the distribution.
    sns.histplot(
        residuals / 1_000,
        bins=50,
        kde=True,
        color="steelblue",
        ax=axes[1],
    )

    axes[1].set_title("Distribution of Residuals")
    axes[1].set_xlabel("Residual Error (£'000)")
    axes[1].set_ylabel("Frequency")


    # ---------------------------------------------------------
    # SAVE OUR DIAGNOSTIC PLOTS
    # ---------------------------------------------------------

    # Adjust the spacing so the two plots are displayed neatly.
    plt.tight_layout()

    # Save the figure so we can include it in our final report
    # or presentation.
    plot_file = figures_dir / "residual_analysis.png"

    plt.savefig(plot_file, dpi=300)

    # Close the figure to release memory.
    plt.close()

    print(f"[SUCCESS] Residual diagnostic plots saved to: {plot_file}")


# Run the evaluation function when this file is executed directly.
if __name__ == "__main__":
    evaluate_pipeline()