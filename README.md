# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 07/10/2025
## NAME: ABISHEK S
## REG NO: 212224240003
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the Netflix stock dataset
file_path = "silver.csv"  # replace with actual path if needed
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data.set_index('Date', inplace=True)

# Use 'USD' price for analysis
usd_prices = data['USD'].dropna()

# Resample to weekly frequency (take mean of each week)
weekly_usd = usd_prices.resample('W').mean()

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(weekly_usd.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# Split into train and test sets (80% training, 20% testing)
train_size = int(len(weekly_usd) * 0.8)
train, test = weekly_usd[:train_size], weekly_usd[train_size:]

# Plot ACF and PACF
fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# Fit AR model with 13 lags
ar_model = AutoReg(train.dropna(), lags=13).fit()

# Make predictions on test set
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot predictions vs test data
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data (Silver USD Price)')
plt.xlabel('Time')
plt.ylabel('USD Price')
plt.legend()
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Plot full data: Train, Test, Predictions
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction (Silver USD Price)')
plt.xlabel('Time')
plt.ylabel('USD Price')
plt.legend()
plt.show()
```
### OUTPUT:

<img width="649" height="490" alt="image" src="https://github.com/user-attachments/assets/f0aecaf9-988b-4b61-aa58-d7dbb4121db5" />
<img width="786" height="733" alt="image" src="https://github.com/user-attachments/assets/6c758339-ba20-427f-a327-eeb91e9d7f05" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.

