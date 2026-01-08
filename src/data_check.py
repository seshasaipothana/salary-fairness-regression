import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/Salary_dataset.csv")

print("\n===== BASIC STRUCTURE =====")
print("Shape (rows, columns):", df.shape)
print("\nColumn types:")
print(df.dtypes)

print("\n===== SAMPLE DATA =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print("Number of duplicates:", df.duplicated().sum())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

# Correlation check
correlation = df["YearsExperience"].corr(df["Salary"])
print("\n===== CORRELATION =====")
print(f"Correlation between experience and salary: {correlation:.3f}")

# Basic EDA plot
plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Data Check)")
plt.savefig("data_check_scatter.png")
plt.clf()

print("\nData check completed successfully.")
