# 📂 1. Importing Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

# 📊 2. Loading the Dataset
df = pd.read_csv('Health_Sleep_Statistics.csv')

# ============================================
# 🧹 3. Exploratory Data Analysis (EDA)
# ============================================

# 3.1 Dataset Overview
print("Size of the dataset : ", df.size)
print("Info of the dataset : ")
df.info()
print("Let's describe the dataset : ")
print(df.describe())

# 3.2 Checking for Null and Duplicate Values
print("The number of null values in the dataset is : ")
print(df.isnull().sum())
print("The number of duplicate values in the dataset is : ", df.duplicated().sum())

# 3.3 Dropping Unnecessary Columns
df.drop(['User ID'], axis=1, inplace=True)

# ============================================
# 📈 4. Age vs Sleep Quality Analysis
# ============================================

# 4.1 Correlation between Age and Sleep Quality
age_sleep_quality_corr = df[['Age', 'Sleep Quality']].corr()
print("The correlation between age and sleep quality : ")
print(age_sleep_quality_corr['Age'])

# 4.2 Bar Plot (Age vs Sleep Quality)
plt.figure(figsize=(12,6))
sns.barplot(x='Age', y='Sleep Quality', data=df, hue='Age', palette='cool', legend=False)
plt.title("Age vs Sleep Quality")
plt.xlabel("Age")
plt.ylabel("Sleep Quality")
plt.show()

# 4.3 Linear Regression Model
age_sleep_quality_model = LinearRegression()
x = df[['Age']]
y = df['Sleep Quality']
age_sleep_quality_model.fit(x, y)

print("Intercept for Age and Sleep Quality is :-")
print(age_sleep_quality_model.intercept_)
print("Slope will be :-")
print(age_sleep_quality_model.coef_)

# 4.4 Plot Actual vs Predicted Values
age_sleep_quality_predictions = age_sleep_quality_model.predict(x)
plt.figure(figsize=(12,6))
sns.scatterplot(x='Age', y='Sleep Quality', data=df, label='Actual Values')
plt.plot(x, age_sleep_quality_predictions, color='green', label='Predicted Values')
plt.legend()
plt.show()

# 4.5 Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4.6 Residual Analysis
''' Residual:-The error (difference) between actual and predicted values
Frequency:-How many data points had that specific residual value (or close to it)'''
residuals = y - age_sleep_quality_predictions
plt.figure(figsize=(10,5))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# ============================================
# 🏃 5. Physical Activity vs Sleep Quality
# ============================================

# 5.1 Visualizing Sleep Quality by Gender
sns.barplot(x='Gender', y='Sleep Quality', data=df)
plt.show()

# 5.2 Visualizing Sleep Quality by Physical Activity
sns.barplot(x='Physical Activity Level', y='Sleep Quality', data=df)
plt.show()

# 5.3 Encoding Physical Activity Level
custom_codes = {'low':1, 'medium':2, 'high':3}
df['Physical Activity Level'] = df['Physical Activity Level'].map(custom_codes)
print("After converting Physical Activity Level into numericals :-")
print(df['Physical Activity Level'].head())

# 5.4 Visualizing Physical Activity by Gender
sns.barplot(x='Gender', y='Physical Activity Level', data=df)
plt.show()

# ============================================
# 📊 6. Gender-Based Statistical Analysis
# ============================================

# 6.1 Separating Male and Female Groups
sleep_quality_males = df[df['Gender'] == 'm']['Sleep Quality']
sleep_quality_females = df[df['Gender'] == 'f']['Sleep Quality']
Physical_activities_males = df[df['Gender'] == 'm']['Physical Activity Level']
Physical_activities_females = df[df['Gender'] == 'f']['Physical Activity Level']

# 6.2 Independent t-Test
t_sleep_quality, p_sleep_quality = stats.ttest_ind(sleep_quality_males, sleep_quality_females)
t_physical_activity, p_physical_activity = stats.ttest_ind(Physical_activities_males, Physical_activities_females)

# 6.3 Reporting t-stat and p-value
'''t-stat:-How different the two groups are, in terms of means
p-value:-The probability that the difference is just random (not real)'''

print(f"t_stats value for Sleep Quality vs Gender : {t_sleep_quality:.2f}")
print(f"p_stats value for Sleep Quality vs Gender : {p_sleep_quality:.2f}")

print(f"t_stats value for Physcal Activity vs Gender : {t_physical_activity:.2f}")
print(f"p_stats value for Physcal Activity vs Gender : {p_physical_activity:.2f}")

# =================================================
# 📈 7. Time of sleep & waking up vs Sleep Quality 
# =================================================

# 7.1 Preview raw time values
print(df['Bedtime'].head())
print(df['Wake-up Time'].head())

# 7.2 Helper to convert HH:MM → decimal hours
def sleep_time_helper(str_time):
    h, m = map(int, str_time.split(":"))
    return h + m / 60

# 7.3 Apply conversion to bedtime and wake-up time
df['Bedtime'] = df['Bedtime'].apply(sleep_time_helper)
df['Wake-up Time'] = df['Wake-up Time'].apply(sleep_time_helper)

# 7.4 Calculate total sleep duration in hours
df['Sleep Duration Hours'] = (df['Wake-up Time'] - df['Bedtime']) % 24

# 7.5 Preview result
print(df['Sleep Duration Hours'].head())
print(df.head())

# =====================================================
# 📊 8. Sleep Duration Hours vs Sleep Quality Analysis
# =====================================================

# 8.1 Bar Plot (Sleep Duration Hours vs Sleep Quality)
plt.figure(figsize=(12,6))
sns.barplot(x='Sleep Duration Hours', y='Sleep Quality', data=df)
plt.title("Sleep Duration Hours vs Sleep Quality")
plt.xlabel("Sleep Duration Hours")
plt.ylabel("Sleep Quality")
plt.show()

# 8.2 Linear Regression Model
sleep_duration_quality_model = LinearRegression()
x = df[['Sleep Duration Hours']]
y = df['Sleep Quality']
sleep_duration_quality_model.fit(x, y)

print("Intercept for Sleep Duration and Sleep Quality is :-")
print(sleep_duration_quality_model.intercept_)
print("Slope will be :-")
print(sleep_duration_quality_model.coef_)

# 8.3 Plot Actual vs Predicted Values
predictions = sleep_duration_quality_model.predict(x)
plt.figure(figsize=(12,6))
sns.scatterplot(x='Sleep Duration Hours', y='Sleep Quality', data=df, label='Actual Values')
plt.plot(x, predictions, color='green', label='Predicted Values')
plt.legend()
plt.show()

# =================================================
# 📈 9. Sleep Disorders vs Sleep Quality 
# =================================================

# 9.1 boxplot
plt.figure(figsize=(12,6))
sns.boxplot(x='Sleep Disorders', y='Sleep Quality', data=df)
plt.show()

# 9.2 Correlation Heatmap for Numerical Features
df_numeric_features= df.select_dtypes(include = ['int64', 'float64'])
print(df_numeric_features.head())
plt.figure(figsize=(12,6))
sns.heatmap(df_numeric_features.corr(), annot=True, cmap="YlGnBu")
plt.show()

# ============================================
# ✅ End of Analysis
# ============================================
