pip install pandas scikit-learn

#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Loading our dataset
data = pd.read_csv('/content/weather_data.csv')

#defining variables
X = data[['TEMPERATURE (°C)']]
y_temp = data['TEMPERATURE (°C)']
y_humidity = data['HUMIDITY (% RH)']

## Spliting the dataset into training and testing sets
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
X_train_humidity, X_test_humidity, y_train_humidity, y_test_humidity = train_test_split(X, y_humidity, test_size=0.2, random_state=42)

# Creating linear regression models
model_temp = LinearRegression()
model_humidity = LinearRegression()

# Training the models
model_temp.fit(X_train_temp, y_train_temp)
model_humidity.fit(X_train_humidity, y_train_humidity)

# Making predictions on the test set
y_pred_temp = model_temp.predict(X_test_temp)
y_pred_humidity = model_humidity.predict(X_test_humidity)

#processing results for MSE
mse_temp = mean_squared_error(y_test_temp, y_pred_temp)
mse_humidity = mean_squared_error(y_test_humidity, y_pred_humidity)

#Printing results for MSE
print(f'Mean Squared Error (Temperature): {mse_temp}')
print(f'Mean Squared Error (Humidity): {mse_humidity}')

#plotting graph for temperature results
plt.scatter(X_test_temp, y_test_temp, color='black')
plt.plot(X_test_temp, y_pred_temp, color='blue', linewidth=3)
plt.title('Temperature Prediction')
plt.xlabel('Temperature')
plt.ylabel('Actual Temperature')
plt.show()

#plotting graph for humidity results
plt.scatter(X_test_humidity, y_test_humidity, color='black')
plt.plot(X_test_humidity, y_pred_humidity, color='red', linewidth=3)
plt.title('Humidity Prediction')
plt.xlabel('Humidity')
plt.ylabel('Actual Humidity')
plt.show()

# Saving the models
joblib.dump(model_temp, 'humidity_model.joblib')
joblib.dump(model_humidity, 'temperature_model.joblib')
