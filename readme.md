# Salary Fairness Analysis using Linear Regression

## 📌 Business Problem
Organizations want to ensure that employee salaries are **fairly aligned with years of experience**.
This project builds a baseline machine learning model to estimate expected salary based on experience
and helps identify potential under- or over-compensation.

---

## 📂 Dataset
- Source: Salary vs Experience dataset (Kaggle)
- Features:
  - `YearsExperience` — total years of professional experience
- Target:
  - `Salary` — annual salary

---

## 🧠 Project Approach

The project is structured to mirror a real-world machine learning workflow:

### 1. Data Validation (`data_check.py`)
- Checked dataset structure and data types
- Verified absence of missing values and duplicates
- Reviewed descriptive statistics for sanity
- Measured correlation between experience and salary
- Generated a scatter plot to visually confirm linearity

### 2. Model Training & Evaluation (`train_model.py`)
- Built a Linear Regression model
- Split data into training and test sets
- Evaluated model performance using:
  - **MAE (Mean Absolute Error)**
  - **R² Score**
- Created diagnostic plots:
  - Regression line vs actual data
  - Residuals vs experience

### 3. Prediction Usage (`predict.py`)
- Trained the model on the full validated dataset
- Generated salary predictions for new experience values
- Designed as a usage-focused script (not analysis)

---

## 📊 Results

- **Correlation (Experience vs Salary):** ~0.98  
- **R² Score:** ~0.90  
- **MAE:** ~6,000 – 7,000  

**Interpretation:**
- Years of experience explains most of the salary variation in this dataset
- Prediction errors are moderate and acceptable for a baseline model
- Residual analysis shows no strong bias patterns

---

## ⚠️ Limitations
- Only one feature (experience) is used
- Dataset is relatively small
- Other real-world factors (role, company, location, negotiation) are not included

This model should be used as a **baseline fairness check**, not a final compensation decision tool.

---

## ▶️ How to Run the Project

### 1. Set up environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
