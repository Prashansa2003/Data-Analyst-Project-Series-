# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load weather data (replace 'weather_data.csv' with your dataset)
weather_df = pd.read_csv('weather_data.csv')

# Data Cleaning
weather_df.dropna(inplace=True)  # Simple example to remove missing values

# Exploratory Data Analysis (EDA)
print(weather_df.describe())

# Visualization of temperature over time
plt.figure(figsize=(10, 5))
plt.plot(weather_df['date'], weather_df['temperature'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Over Time')
plt.show()

# Statistical Analysis
correlation_matrix = weather_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Predictive Modeling
# Example: Predicting temperature based on other features
features = weather_df[['humidity', 'pressure', 'wind_speed']]
target = weather_df['temperature']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualization of Predictions vs Actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.show()

