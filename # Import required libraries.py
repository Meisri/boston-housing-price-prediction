# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing  # Use California Housing dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
california_housing = fetch_california_housing()

# Create a DataFrame from the dataset
california_df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_df['Price'] = california_housing.target

# Display first few rows of the dataset
print(california_df.head())

# Data Preprocessing
# Check for missing values
print(california_df.isnull().sum())

# Split the data into features and target
X = california_df.drop('Price', axis=1)
y = california_df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict using the Linear Regression model
y_pred_lr = lr_model.predict(X_test)

# Calculate performance metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - Mean Squared Error: {mse_lr}")
print(f"Linear Regression - R-squared: {r2_lr}")

# Create Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict using the Ridge Regression model
y_pred_ridge = ridge_model.predict(X_test)

# Calculate performance metrics for Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression - R-squared: {r2_ridge}")

# Visualize the predictions vs actual prices for Linear Regression
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Prediction')
plt.title("Linear Regression: Predictions vs Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.show()

# Visualize the predictions vs actual prices for Ridge Regression
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_ridge, color='green', label='Ridge Regression Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Prediction')
plt.title("Ridge Regression: Predictions vs Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.show()

# Display correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(california_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of California Housing Dataset")
plt.show()
