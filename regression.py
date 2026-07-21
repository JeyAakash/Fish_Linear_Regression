# Fish Weight Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Fishers maket.csv")

# Convert Species (Text → Number)
df["Species"] = LabelEncoder().fit_transform(df["Species"])

# Input (Features) and Output (Target)
X = df.drop("Weight", axis=1)
y = df["Weight"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Weight
y_pred = model.predict(X_test)

# Evaluate Model
print("\nModel Performance")
print("-" * 30)
print("MAE      :", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE      :", round(mean_squared_error(y_test, y_pred), 2))
print("RMSE     :", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
print("R² Score :", round(r2_score(y_test, y_pred), 4))

# Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2)
})

print("\nFirst 10 Predictions")
print(result.head(10))

# Regression Equation
print("\nRegression Equation")
print("-" * 30)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print("Intercept:", round(model.intercept_, 2))

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Actual vs Predicted Fish Weight")
plt.grid(True)
plt.show()