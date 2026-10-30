# UK Housing Price Analysis & Machine Learning

## Project Overview

This project explores the London housing market using property data to understand the factors influencing house prices and develop a machine learning model capable of estimating property values.

The project follows an end-to-end data analysis and machine learning workflow, starting with data cleaning and exploratory data analysis (EDA), followed by feature selection, model training, prediction and evaluation.

Our main objectives are:

1. Analyse historical housing market trends and understand how prices vary over time, location and property type.
2. Develop a supervised machine learning regression model to estimate property prices based on characteristics such as floor area, number of bedrooms, bathrooms, property type and location.

## Research Questions

The project aims to answer two main questions:

- **Market trends:** How might house prices change between now and next year across different property types and locations?
- **Price prediction:** Given a property's size, location, number of bedrooms and other characteristics, what price would it likely sell for?

## Dataset

The project uses London housing data containing property characteristics and estimated property values.

The original dataset includes information such as:

| Column | Description |
|---|---|
| `price` | Property price target used by the model |
| `floorAreaSqM` | Property floor area in square metres |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `livingRooms` | Number of living rooms |
| `propertyType` | Type of property, such as flat or detached |
| `tenure` | Property ownership type, such as freehold or leasehold |
| `outcode` | Outward part of the postcode, representing location |

The cleaned dataset contains **417,436 property records**.

*Note: The target variable is mapped from `saleEstimate_currentPrice` in the original dataset. Therefore, the model estimates the supplied property valuation rather than directly predicting verified completed sale prices.*



## Methodology

The project follows a structured machine learning workflow.

### 1. Data Cleaning and Preparation

**Script:** `src/data_cleaning.py`

The initial ETL pipeline prepares the raw dataset for analysis and modelling.

Key operations include:

- Removing duplicate records.
- Removing records with missing or unrealistic prices.
- Filtering physically implausible property measurements.
- Standardising categorical values.
- Removing unnecessary valuation fields to reduce potential data leakage.
- Exporting the cleaned dataset for subsequent stages.

**Output:** `data/processed/cleaned_london_house_prices.csv`

### 2. Exploratory Data Analysis (EDA)

**Notebook:** `notebooks/01_data_exploration.ipynb`

Exploratory analysis is performed to understand the dataset and identify relationships between property characteristics and prices.

This includes:

- Inspecting dataset structure, data types and missing values.
- Analysing price distributions and skewness.
- Comparing property prices across property types and tenure.
- Investigating relationships between numerical variables and prices.
- Checking correlations and potential multicollinearity.
- Reviewing initial Linear Regression assumptions.

A log transformation of prices is also explored to understand the effect of skewness.

### 3. Feature Selection and Model Preparation

**Script:** `src/train_model.py`

The target and predictor variables are defined.

**Target variable (y):**

- `price`

**Numerical features (X):**

- `floorAreaSqM`
- `bedrooms`
- `bathrooms`
- `livingRooms`

**Categorical features (X):**

- `propertyType`
- `tenure`
- `outcode`

The dataset is split into:

- 80% training data.
- 20% testing data.

A fixed random state (`42`) is used to ensure reproducibility.

### 4. Preprocessing and Model Training

A Scikit-learn Pipeline combines preprocessing and model training.

The preprocessing includes:

| Component | Purpose |
|---|---|
| `SimpleImputer(strategy='median')` | Fills missing numerical values using the median. |
| `SimpleImputer(strategy='most_frequent')` | Fills missing categorical values using the most frequent category. |
| `OneHotEncoder` | Converts categorical variables into numerical features. |
| `ColumnTransformer` | Applies different preprocessing operations to numerical and categorical columns. |
| `LinearRegression` | Trains the baseline regression model. |
| `Pipeline` | Combines preprocessing and model training into one workflow. |

The trained pipeline is saved using Joblib.

**Outputs:**

- `models/baseline_linear_model.joblib`
- `data/processed/test_set_eval.csv`

### 5. Model Evaluation

**Script:** `src/evaluate_model.py`

The saved model is loaded and evaluated against the held-out test dataset.

Predictions are generated using `.predict()` and compared with the actual target values.

The following evaluation metrics are calculated:

- **R²:** Measures the proportion of variation in the target explained by the model.
- **MAE:** Measures the average absolute difference between predicted and actual values.
- **RMSE:** Measures prediction error while penalising larger errors more heavily.

### 6. Residual Analysis

Residuals are calculated as:

`Residual = Actual Price - Predicted Price`

Two diagnostic visualisations are generated:

1. **Residuals vs Fitted Values:** Investigates error patterns and whether residual variance is consistent across predicted prices.
2. **Distribution of Residuals:** Examines the distribution of prediction errors and their deviation from normality.

The diagnostic figure is exported to:

`reports/figures/residual_analysis.png`

## Baseline Model Results

The initial Linear Regression model produced the following results on the held-out test dataset:

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.7060 |
| Mean Absolute Error (MAE) | £252,152.42 |
| Root Mean Squared Error (RMSE) | £484,528.47 |
| Training records | 333,948 |
| Testing records | 83,488 |

### Interpretation

The baseline model explains approximately 70.6% of the variation in the target property values.

However, the relatively high MAE and RMSE indicate that prediction errors remain substantial.

Residual analysis also identified:

- Increasing error variance at higher predicted prices, suggesting heteroscedasticity.
- A right-skewed residual distribution, indicating some substantial underpredictions.

These findings highlight opportunities for further model improvement.

## Technologies and Libraries

The project uses Python and the following libraries:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, regression and evaluation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Joblib | Saving and loading trained models |
| Jupyter Notebook | Exploratory data analysis |

## Installation and Setup

### 1. Clone the repository

```bash
git clone <https://github.com/salmamxz/uk-housing-price-analysis>
cd uk-housing-price-analysis
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

Run the scripts in the following order.

**Step 1: Clean the raw data**

```bash
python src/data_cleaning.py
```

**Step 2: Explore the data**

Open the notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Step 3: Train the model**

```bash
python src/train_model.py
```

**Step 4: Evaluate the model**

```bash
python src/evaluate_model.py
```

*Note: The original dataset must be placed in the expected raw data location, and the input path in the cleaning script must match its filename.*

## Future Improvements

The baseline Linear Regression model provides a starting point for further experimentation.

Planned improvements include:

- Applying a log transformation to the target variable.
- Comparing Linear Regression with Ridge Regression.
- Experimenting with Random Forest Regression to capture non-linear relationships.
- Investigating outliers and large prediction errors.
- Comparing evaluation metrics across different models.
- Exploring time-based train/test splits to assess performance on later observations.
- Integrating historical market trends to investigate changes in property prices over time.

The aim is to identify a more reliable model while maintaining an appropriate evaluation methodology.

## Team Contribution

This project was developed collaboratively, combining exploratory data analysis, housing market research and machine learning.

The work is organised into separate stages to support collaboration, reproducibility and maintainability.

## Data Attribution

Where applicable, the original dataset's licensing terms and attribution requirements should be followed. Any HM Land Registry data used in the project should retain the required attribution:

"Contains HM Land Registry data © Crown copyright and database right. This data is licensed under the Open Government Licence v3.0."

---

**Project:** UK Housing Price Analysis & Machine Learning

**Focus:** Exploratory Data Analysis, Regression Modelling and Property Price Prediction

**Status:** Baseline model completed; model improvement and comparison in progress.