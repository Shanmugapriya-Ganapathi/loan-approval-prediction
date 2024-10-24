import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified file path
file_path = r'C:\Users\Surya M\Downloads\loan-prediction.csv'
data = pd.read_csv(file_path)

# Display first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Check the summary of the dataset
print("\nSummary of the dataset:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Visualizing distribution of ApplicantIncome
plt.figure(figsize=(10, 6))
sns.histplot(data['ApplicantIncome'], bins=30, kde=True)
plt.title('Distribution of Applicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Select only numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=[np.number])

# Visualizing the correlation matrix
plt.figure(figsize=(12, 8))
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()

# Exploring Gender vs Loan Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='LoanAmount', data=data)
plt.title('Loan Amount by Gender')
plt.grid()
plt.show()

# Exploring Loan Amount by Education
plt.figure(figsize=(10, 6))
sns.boxplot(x='Education', y='LoanAmount', data=data)
plt.title('Loan Amount by Education')
plt.grid()
plt.show()
