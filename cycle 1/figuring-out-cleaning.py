# Importing all necessary libraries
import pandas as pd                 # For data manipulation
import numpy as np                  # For numerical computations
import seaborn as sns               # For advanced visualizations
import matplotlib.pyplot as plt     # For basic plotting
import warnings

# Settings to improve readability
warnings.filterwarnings("ignore")           #hides warning messages
sns.set(style="whitegrid")                  #sets a clean background for plots
pd.set_option('display.max_columns', None)  #shows all columns when df.hea() is typed. won't hide the middles part with ...


# Load the dataset
df = pd.read_csv("mental_health_workplace.csv")  #to read the csv file

# Preview first few rows
df.head() # shows first 5 rows specifically    



# Shape of the dataset
print("Total Rows:", df.shape[0]) # first value touple (ie rows)         [doubt --> (so i guess shape refers to accessing the data as touple...cross check!)]    [answer: apparently yes]
print("Total Columns:", df.shape[1]) # second valiue touple (ie columns) 

# Check data types and non-null counts
df.info()  # shows column names, data types and non-null counts  [then y r we printing the colums in prev line ?....maybe we want the system to remember it this time..last time we wanted to know...we'll see] [answer:ok so the above 2 lines are apparently like asking how big is my data...this one gives and overall structure and yes also stores it...yea..it was kinda obv..dumbass] 

# Check summary statistics for numeric columns
df.describe() #[gives mean, standard deviation etc]


# Check total missing values per column
df.isnull().sum().sort_values(ascending=False)
# ok, we need to break that down...ok so :
  # df.isnull()                   # Returns True for NaN cells, False otherwise
  # df.isnull().sum()             # Counts how many Trues (i.e., NaNs) per column
  #.sort_values(ascending=False)  # Sorts the result so the columns with MOST missing values come first]


# Handle numeric columns (int/float)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols: #finds numeric columns
    median_val = df[col].median() #finds median
    df[col].fillna(median_val, inplace=True)  # Replaces missing values with median
    print(f"Filled missing values in {col} with median: {median_val}")

# Handle categorical columns (object/string)
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)    # Replaces missing values with most frequent value.....ok but that somehow feels wrong...but...ig that's how it is]
    print(f"Filled missing values in {col} with mode: {mode_val}")



# Plot boxplots for key numeric features
features = ['Work_hrs_per_day', 'Sleep_hours', 'Stress_level'] 
for col in features:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot for {col}")
    plt.show()


# writing a function to which removes outliners
def remove_outliers_iqr(df, col): # (df, col) meand it takes in df a data frame-wrork and col means a particular frame work
    Q1 = df[col].quantile(0.25) # um i didnt quite understand this part but apparently it takes 25 percentile of the data...and 25% of the data lies below this...um..yea..idk
    Q3 = df[col].quantile(0.75) # here 75% lies below this value 
    IQR = Q3 - Q1 #ok so this is called the interquartile range...it is the middle ie 50% of the data...ookayy??...tells how to spread out the central data apparently...what's going on ?..
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    original_size = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed = original_size - df.shape[0]
    print(f"{removed} outliers removed from '{col}'")
    return df

# Remove outliers from important features
for col in ['Work_hrs_per_day', 'Sleep_hours', 'Stress_level']:
    df = remove_outliers_iqr(df, col)



# Burnout counts
sns.countplot(x='Burnout', data=df)
plt.title("Burnout Distribution")
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()



# Boxplot of Sleep_hours vs Burnout
sns.boxplot(x='Burnout', y='Sleep_hours', data=df)
plt.title("Sleep Hours by Burnout")
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Sleep Hours")
plt.show()



# Scatterplot of Stress vs Sleep colored by Burnout
sns.scatterplot(x='Sleep_hours', y='Stress_level', hue='Burnout', data=df, palette='coolwarm')
plt.title("Stress vs Sleep (Colored by Burnout)")
plt.xlabel("Sleep Hours")
plt.ylabel("Stress Level")
plt.legend(title="Burnout")
plt.show()



plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()