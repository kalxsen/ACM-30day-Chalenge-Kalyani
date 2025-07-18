# Importing all necessary libraries
import pandas as pd                 # For data manipulation
import numpy as np                  # For numerical computations
import seaborn as sns               # For advanced visualizations
import matplotlib.pyplot as plt     # For basic plotting
import warnings

# Settings to improve readability
warnings.filterwarnings("ignore")           #hides warning messages
sns.set(style="whitegrid")                  #sets a clean background for plots
pd.set_option('display.max_columns', None)  #shows all columns when df.head() is typed. won't hide the middles part with ...


# Load the dataset
df = pd.read_csv("mental_health_workplace_survey.csv")  #to read the csv file

# Preview first few rows
df.head() # shows first 5 rows specifically    



# Shape of the dataset
print("Total Rows:", df.shape[0]) # first value touple (ie rows)
print("Total Columns:", df.shape[1]) # second valiue touple (ie columns) 

# Check data types and non-null counts
df.info()  # shows column names, data types and non-null counts 

# Check summary statistics for numeric columns
df.describe() #[gives mean, standard deviation etc]


# Check total missing values per column
df.isnull().sum().sort_values(ascending=False)

# Handle numeric columns (int/float)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)  # Replaces missing values with median
    print(f"Filled missing values in {col} with median: {median_val}")

# Handle categorical columns (object/string)
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)    # Replaces missing values with most frequent value
    print(f"Filled missing values in {col} with mode: {mode_val}")



# Plot boxplots for key numeric features
features = ['WorkHoursPerWeek', 'SleepHours', 'StressLevel']
for col in features:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot for {col}")
    plt.show()



def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    original_size = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed = original_size - df.shape[0]
    print(f"{removed} outliers removed from '{col}'")
    return df

# Remove outliers from important features
for col in ['WorkHoursPerWeek', 'StressLevel', 'SleepHours']:
    df = remove_outliers_iqr(df, col)



# Burnout counts
sns.countplot(x='BurnoutLevel', data=df)
plt.title("Burnout Distribution")
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()



# Boxplot of Sleep_hours vs Burnout
sns.boxplot(x='BurnoutLevel', y='SleepHours', data=df)
plt.title("Sleep Hours by Burnout")
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Sleep Hours")
plt.show()



# Scatterplot of Stress vs Sleep colored by Burnout
sns.scatterplot(x='SleepHours', y='StressLevel', hue='BurnoutLevel', data=df, palette='coolwarm')
plt.title("Stress vs Sleep (Colored by Burnout)")
plt.xlabel("Sleep Hours")
plt.ylabel("Stress Level")
plt.legend(title="Burnout")
plt.show()



plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

