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
# the column heading in the data was diffrent from original code but now i have corrected the column names in the code keeping the name in the dataset the same 
df = pd.read_csv("mental_health_workplace_survey.csv")  #to read the csv file

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
features = ['WorkHoursPerWeek', 'SleepHours', 'StressLevel']
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
    lower_bound = Q1 - 1.5 * IQR # lower oultiner apparenly...how did they come up with 1.5 ?? oh, ok it's commonly used multiplier for IQR (Interquatrile Range)...i got more questions but nope..gotta move on...so we multiply the mid range with 1.5 (y multiply is it stretching the range or something?) then we take that part off of the 25 percentile...huh?..ohhh..gives lower bound...and 1.5 multiplied gives a less strict division of data to take as lower bound
    upper_bound = Q3 + 1.5 * IQR # similarly for upper bound we get a value above the normal distribution range...

    #ok..finally figured it out...Q1 was 25 percentile Q3 was my 75 percentile and Q3-Q1 gave me the medial range...multiplying the median range with a multiplier gave me a less strict range for median values...now we found the actual lower bound by Q1-1.5*IQR(interquatrile range ie that 50% thingy)...it would usually be a value less than Q1 (ig we are bassically widening the range we consider further by bringing the bar down futher incase of lower and furthur up in case of upper bounds respectively)....similary calculate upperbound with the formula above...it would give a value above the upper bound....now we ask it to take the values above lower bound and those below upper bound ig) 

    original_size = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)] # so this checks if the purticular row has values higher than lower bound and lower than upper bound....if yes then it stores it...
    removed = original_size - df.shape[0] #finding the number of rows that were removed
    print(f"{removed} outliers removed from '{col}'") # just a print statement that tells how many values were removed from that particular column
    return df

# Remove outliers from important features
for col in ['WorkHoursPerWeek', 'StressLevel', 'SleepHours']: #defined the columns from which we wanted to remove outliners from
    df = remove_outliers_iqr(df, col) # ran that function to remove it.



# Burnout counts
sns.countplot(x='BurnoutLevel', data=df)
plt.title("Burnout Distribution")  # like matlab....nothing new....but the type of graph is diffrent
plt.xlabel("Burnout (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()



# Boxplot of Sleep_hours vs Burnout
sns.boxplot(x='BurnoutLevel', y='SleepHours', data=df)
plt.title("Sleep Hours by Burnout")   # like matlab....nothing new....but the type of graph is diffrent
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
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm') # we changed it here a little cause male was not a numeric value and the code was trying to process it but now it only takes numeric values...
plt.title("Feature Correlation Heatmap")
plt.show()