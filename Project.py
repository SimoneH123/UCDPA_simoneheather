# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Reading the data into a data frame
df = pd.read_csv('HR-Employee-Attrition.csv')

# Understanding the structure of the dataset
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())

# Count missing values
missing_values_count = df.isnull().sum()
print(missing_values_count)

# Drop Duplicates
drop_duplicates = df.drop_duplicates()
print(df.shape, drop_duplicates.shape)

# Define lists of those who have left (attrition) and those who have stayed (retention)
attrition = df[df["Attrition"] == "Yes"]
retention = df[df["Attrition"] == "No"]
print(attrition)
print(retention)

# Sort Attrition by Age
sort_attrition = attrition.sort_values(by='Age')
print(sort_attrition)

# Group Attrition by Department
group_attrition = attrition.groupby(["Department"])
print(group_attrition)

# Create index and values for Attrition
labels = df.Attrition.value_counts().index
sizes = df.Attrition.value_counts().values
colors = ["green", "salmon"]
print(labels)
print(sizes)

# Create numpy array to include all satisfaction factors
data = np.array([["EmployeeNumber"], ["Age"], ['Department'], ["Attrition"], ["EnvironmentSatisfaction"],
                 ["JobSatisfaction"], ["RelationshipSatisfaction"]])
print(data)

# Convert Attrition column to numeric value
Attrition_no = pd.get_dummies(attrition.Attrition, drop_first=True)
print(Attrition_no)

# Combine satisfaction ratings
satisfaction = df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']
print(satisfaction)

# Visualisations

# Pie chart of Attrition Rate
plt.figure()
plt.pie(sizes, labels=labels, colors=colors, autopct="%.1f%%")
plt.title("Employee Attrition Rate", color='green', fontsize=20)
plt.show()

# Bar Chart of Attrition by Number of Employees
sns.countplot(data=df, x='Attrition')
plt.title('Attrition by Number of Employees')
plt.show()

# Age profiling of All Staff
plt.xlabel('Age')
plt.ylabel('No of Employees')
plt.title('Age Profile of Employees')
plt.hist(df['Age'], bins=20)
plt.show()

# Gender profiling of All Staff
plt.xlabel('Gender')
plt.ylabel('No of Employees')
plt.title('Gender Profile of Employees')
plt.hist(df['Gender'], bins=3)
plt.show()

# Service profiling of All Staff
sns.countplot(data=df, x='YearsAtCompany')
plt.title('Service in Years at the Company')
plt.xticks([5, 10, 15, 20, 25, 30, 35, 40])
plt.show()

# Employee By Field of Education
sns.countplot(data=df, x='EducationField')
plt.title('Employees by Field of Education')
plt.xticks(rotation=45)
plt.show()

# Attrition by Department, Gender, Age & Job Role
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.countplot(data=attrition, x='Department', ax=ax[0, 0])
sns.countplot(data=attrition, x='Gender', ax=ax[0, 1])
sns.countplot(data=attrition, x='Age', ax=ax[1, 0])
plt.xticks([20, 30, 40, 50, 60])
sns.countplot(data=attrition, x='JobRole', ax=ax[1, 1])
plt.xticks(rotation=45)
plt.show()
