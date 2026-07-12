# Project Infidelity Regression Analysis
**Model:** Lasso Linear Regression (L1-Regularized)

## Project Description
An analytical machine learning pipeline designed to predict the frequency of extramarital affairs based on demographic, social, and relationship features from the Fair's Affairs dataset. The project isolates active cases by filtering out zero values and implements standardized column transformations to optimize regularized feature selection and prediction accuracy.

## Data Preprocessing
* Target Isolation: Removed zero values from the target variable to avoid denominator inflation and establish realistic percentage errors.
* Feature Scaling: Applied standard normalization via StandardScaler across all demographic, categorical dummy, and interaction ratio features.
* Column Transformation: Standardized all feature columns selectively while maintaining the integrity and original scale of the target variable.

## Model Evaluation Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso

# Setup preprocessor for selective scaling
preprocessor = ColumnTransformer(
    transformers=[('scale_features', StandardScaler(), cols_to_standardize)],
    remainder='passthrough'
)

# Build pipeline with Lasso regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lasso', Lasso(alpha=0.1))
])
```

## Performance Performance Metrics
* Evaluated via Mean Absolute Error (MAE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and R-squared (R2) Score.
* Stability checked via continuous evaluation on independent training and validation test splits.
