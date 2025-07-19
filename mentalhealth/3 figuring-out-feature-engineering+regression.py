# CODE FOR CLEANING THE DATE (TILL APPROX LINE 82)

import pandas as pd                
import numpy as np                 
import seaborn as sns              
import matplotlib.pyplot as plt     
import warnings

warnings.filterwarnings("ignore")          
sns.set(style="whitegrid")                  
pd.set_option('display.max_columns', None)  

df = pd.read_csv("mental_health_workplace_survey.csv")
df.head()  

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

df.info()
df.describe()
df.isnull().sum().sort_values(ascending=False)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True) 
    print(f"Filled missing values in {col} with median: {median_val}")

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)    
    print(f"Filled missing values in {col} with mode: {mode_val}")

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

for col in ['WorkHoursPerWeek', 'StressLevel', 'SleepHours']:
    df = remove_outliers_iqr(df, col)

sns.countplot(x='BurnoutLevel', data=df)
plt.title("Burnout Distribution")
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

sns.boxplot(x='BurnoutLevel', y='SleepHours', data=df)
plt.title("Sleep Hours by Burnout")
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Sleep Hours")
plt.show()

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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FEATURE ENGINEERING AND REGRESSION CYCLE

#feature engineering is done to make the data learnable for the machine...
#regression is to understand patterns between inputs and outputs and give proper outputs

#ordinal encoding:
#When working with any data related to ranking something with non-numerical categories 

#one-hot encoding : 