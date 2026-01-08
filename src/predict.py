import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/raw/Salary_dataset.csv")

# Separate feature and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Train model on full dataset
model = LinearRegression()
model.fit(X, y)

# ---- Prediction section ----
years_of_experience = 5  # change this value to test
input_df = pd.DataFrame(
    {"YearsExperience": [years_of_experience]}
)
predicted_salary = model.predict(input_df)


print(
    f"Predicted salary for {years_of_experience} years of experience: "
    f"{predicted_salary[0]:.2f}"
)
