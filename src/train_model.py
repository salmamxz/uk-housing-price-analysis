# This file handles three stages of our machine learning workflow:
# Step 3: Choose the features (X) and target (y).
# Step 5: Split the data into training and testing sets.
# Step 6: Prepare the data, build and train our Linear Regression model.
#
# We also save the trained model and test data so we can use them
# later in evaluate_model.py without having to train everything again.


from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_and_train_model():

    # First, find the main project folder.
    # This allows us to build file paths without hardcoding
    # the full location on our computer.
    project_root = Path(__file__).resolve().parent.parent

    # Define where our cleaned dataset and model will be stored.
    data_path = (
        project_root / "data" / "processed" / "cleaned_london_house_prices.csv"
    )
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    # Create the models folder if it doesn't already exist.
    models_dir.mkdir(exist_ok=True)

    # Load the cleaned dataset into a Pandas DataFrame.
    print(f"[INFO] Loading processed dataset from: {data_path}")
    df = pd.read_csv(data_path)


    # ---------------------------------------------------------
    # STEP 3: CHOOSE OUR FEATURES (X) AND TARGET (y)
    # ---------------------------------------------------------

    # The target is what we want our model to predict.
    target = "price"

    # These are our numerical features.
    # We use a list comprehension to select the columns that
    # actually exist in our dataset.
    numeric_features = [
        col
        for col in ["floorAreaSqM", "bedrooms", "bathrooms", "livingRooms"]
        if col in df.columns
    ]

    # These are our categorical features.
    # They describe the type of property and its location.
    categorical_features = [
        col for col in ["propertyType", "tenure", "outcode"] if col in df.columns
    ]

    # Combine both groups into one feature list.
    # X contains the information we give the model.
    # y contains the price we want the model to predict.
    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]
    y = df[target]

    print(f"[INFO] Predictors ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Target: {target}")


    # ---------------------------------------------------------
    # STEP 5: SPLIT INTO TRAINING AND TESTING DATA
    # ---------------------------------------------------------

    # We split our data into 80% for training and 20% for testing.
    #
    # Training data teaches the model patterns in the data.
    # Testing data allows us to check how well it performs
    # on properties it hasn't seen before.
    #
    # random_state=42 makes sure we get the same split
    # every time we run the script.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(
        f"[INFO] Training set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows"
    )


    # ---------------------------------------------------------
    # STEP 6: BUILD OUR PREPROCESSING PIPELINE
    # ---------------------------------------------------------

    # Our dataset contains both numerical and categorical data.
    # They need different preparation before we can train the model.
    #
    # ColumnTransformer allows us to apply different operations
    # to different groups of columns.

    preprocessor = ColumnTransformer(
        transformers=[

            # NUMERICAL FEATURES:
            # Some properties have missing values for size or rooms.
            # We fill those missing values using the median.
            #
            # The median is less affected by extreme values
            # than the mean.
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),

            # CATEGORICAL FEATURES:
            # First, fill missing categories with the most common value.
            # Then, use OneHotEncoder to convert text categories
            # into numerical columns that the model can understand.
            #
            # handle_unknown="ignore" means the model can handle
            # categories it didn't encounter during training.
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    # Now we combine preprocessing and Linear Regression
    # into one pipeline.
    #
    # This means the data is automatically prepared before
    # being passed to the regression model.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # Train the complete pipeline using our training data.
    # The model learns the relationships between property features
    # and the target prices.
    print("[INFO] Fitting Linear Regression pipeline...")
    pipeline.fit(X_train, y_train)


    # ---------------------------------------------------------
    # SAVE OUR MODEL AND TEST DATA
    # ---------------------------------------------------------

    # Save the trained pipeline using Joblib.
    # We can load it later without training the model again.
    joblib.dump(pipeline, models_dir / "baseline_linear_model.joblib")

    # Save the test features alongside their actual target values.
    # This allows our evaluation script to compare predictions
    # against the original values using exactly the same test set.
    test_data = X_test.copy()
    test_data["actual_price"] = y_test

    test_data.to_csv(processed_dir / "test_set_eval.csv", index=False)

    print(f"[SUCCESS] Trained model saved to: {models_dir / 'baseline_linear_model.joblib'}")
    print(f"[SUCCESS] Held-out test set saved to: {processed_dir / 'test_set_eval.csv'}")


# This makes sure the training function only runs
# when we execute this file directly.
if __name__ == "__main__":
    build_and_train_model()