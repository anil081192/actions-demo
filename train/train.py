import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import subprocess

# -------------------------
# Path Setup
# -------------------------
# Get the current script directory (train/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Project root (one level up from 'train')
project_root = os.path.dirname(current_dir)

# Paths
data_path = os.path.join(project_root, "data", "Advertising.csv")
model_dir = os.path.join(project_root, "model")
os.makedirs(model_dir, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(data_path)

print("Initial Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------
# Feature Engineering
# -------------------------
df["TV_Radio"] = df["TV"] * df["Radio"]
df["TV_Newspaper"] = df["TV"] * df["Newspaper"]
df["Radio_Newspaper"] = df["Radio"] * df["Newspaper"]
df["TV_Radio_Newspaper"] = df["TV"] * df["Radio"] * df["Newspaper"]
df["TV_squared"] = df["TV"] ** 2
df["Radio_squared"] = df["Radio"] ** 2
df["Newspaper_squared"] = df["Newspaper"] ** 2

print("\nData Head after Feature Engineering:")
print(df.head())

# -------------------------
# Train/Test Split
# -------------------------
X = df.drop("Sales", axis=1)
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# -------------------------
# Train Models
# -------------------------
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
print("\nModels trained successfully.")

# -------------------------
# Evaluate Models
# -------------------------
models = {
    "Linear Regression": linear_model,
    "Ridge Regression": ridge_model,
    "Lasso Regression": lasso_model,
}

print("\nModel Evaluation Metrics:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Metrics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")
    print("-" * 20)

# -------------------------
# Save Best Model
# -------------------------
best_model = linear_model
filename = os.path.join(model_dir, "best_regression_model.pkl")
with open(filename, "wb") as f:
    pickle.dump(best_model, f)
print(f"\nBest model saved as {filename}")
