# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the weather dataset
weather_df = pd.read_csv('weather_data.csv')

# Display the first few rows of the dataset
print(weather_df.head())

# Check for missing values
print(weather_df.isnull().sum())

# Handle missing values (if any)
# Option 1: Remove rows with missing values
# weather_df_cleaned = weather_df.dropna()

# Option 2: Fill missing values with mean or median
# weather_df_filled = weather_df.fillna(weather_df.mean())

# For simplicity, use median to fill missing values for this example
weather_df['temperature'] = weather_df['temperature'].fillna(weather_df['temperature'].median())
weather_df['humidity'] = weather_df['humidity'].fillna(weather_df['humidity'].median())

print(weather_df.isnull().sum())

# Detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in the temperature column
outliers_temp = detect_outliers_iqr(weather_df, 'temperature')
print(outliers_temp)

# Option 1: Remove outliers
weather_df_no_outliers = weather_df[~weather_df.isin(outliers_temp)].dropna()

# Option 2: Cap the outliers
weather_df['temperature'] = np.where(weather_df['temperature'] > weather_df['temperature'].quantile(0.95),
                                     weather_df['temperature'].quantile(0.95),
                                     np.where(weather_df['temperature'] < weather_df['temperature'].quantile(0.05),
                                              weather_df['temperature'].quantile(0.05),
                                              weather_df['temperature']))

print(weather_df.describe())

# Check data types
print(weather_df.dtypes)

# Convert date column to datetime format
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Standardize units (if necessary)
# For example, convert wind speed from km/h to m/s
weather_df['wind_speed_m_s'] = weather_df['wind_speed_kmh'] / 3.6

# Display the cleaned dataset
print(weather_df.head())

# Save the cleaned dataset to a new CSV file
weather_df.to_csv('cleaned_weather_data.csv', index=False)
