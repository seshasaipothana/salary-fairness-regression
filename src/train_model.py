import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("data/raw/Salary_dataset.csv")

# 2. Separate feature and target
X = df[["YearsExperience"]]
y = df["Salary"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# 7. Plot: Regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.savefig("regression_line.png")
plt.clf()

# 8. Plot: Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.savefig("actual_vs_predicted.png")
plt.clf()

# 9. Plot: Residuals
residuals = y_test - y_pred
plt.scatter(X_test, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Years of Experience")
plt.ylabel("Residuals")
plt.title("Residuals vs Experience")
plt.savefig("residuals.png")
