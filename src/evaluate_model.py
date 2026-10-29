#Handles Steps 7, 8, and 9:
#- Loads the trained pipeline and held-out test data
#- Generates predictions
#- Calculates MAE, RMSE, and R²
#- Performs residual diagnostics and exports plots to reports/figures/


from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_pipeline():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "baseline_linear_model.joblib"
    test_path = project_root / "data" / "processed" / "test_set_eval.csv"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists() or not test_path.exists():
        print("[ERROR] Missing model or test dataset. Run src/train_model.py first.")
        return

    print("[INFO] Loading model and test set...")
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    y_test = test_df["actual_price"]
    X_test = test_df.drop(columns=["actual_price"])

    # Step 7: Make predictions
    print("[INFO] Generating predictions on test set...")
    y_pred = model.predict(X_test)

    # Step 8: Evaluate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 45)
    print("         MODEL PERFORMANCE METRICS")
    print("=" * 45)
    print(f"Mean Absolute Error (MAE) : £{mae:,.2f}")
    print(f"Root Mean Squared Error   : £{rmse:,.2f}")
    print(f"R² Score                  : {r2:.4f}")
    print("=" * 45 + "\n")

    # Step 9: Residual diagnostics
    residuals = y_test - y_pred

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Residuals vs Predicted Prices (Homoscedasticity check)
    # Subsample 5,000 points to keep plot clear and responsive
    plot_idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)

    sns.scatterplot(
        x=y_pred[plot_idx] / 1_000,
        y=residuals.iloc[plot_idx] / 1_000,
        alpha=0.3,
        color="crimson",
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Residuals vs. Fitted Values (Sample)")
    axes[0].set_xlabel("Predicted Price (£'000)")
    axes[0].set_ylabel("Residual (£'000)")

    # 2. Distribution of Residuals (Normality check)
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

    plt.tight_layout()
    plot_file = figures_dir / "residual_analysis.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"[SUCCESS] Residual diagnostic plots saved to: {plot_file}")


if __name__ == "__main__":
    evaluate_pipeline()