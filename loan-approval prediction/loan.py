import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = r"C:\Users\hp\Downloads\loan-prediction.csv"
data = pd.read_csv(file_path)
print("First 5 rows of the dataset:")
print(data.head())
print("\nSummary of the dataset:")
print(data.describe())
print("\nMissing values in each column:")
print(data.isnull().sum())
plt.figure(figsize=(10, 6))
sns.histplot(data['ApplicantIncome'], bins=30, kde=True)
plt.title('Distribution of Applicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')
plt.grid()
plt.show()
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='LoanAmount', data=data)
plt.title('Loan Amount by Gender')
plt.grid()
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x='Education', y='LoanAmount', data=data)
plt.title('Loan Amount by Education')
plt.grid()
plt.show()
