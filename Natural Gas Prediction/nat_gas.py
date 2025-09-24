import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

nat_gas_data = pd.read_csv('C:\\Users\\josse\\OneDrive\\CS\\Projet\\JP MORGAN\\Natural Gas Prediction\\Nat_Gas.csv')

t = np.arange(len(nat_gas_data))

a = np.polyfit(t, nat_gas_data['Prices'], 1)
drift = a[0] * t + a[1]
y = nat_gas_data['Prices'] - drift

N = 1

yf = np.fft.fft(y)
T = 1.0
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

sorted_indices = np.argsort(np.abs(yf[:N//2]))[::-1][0:6]
dominant_freqs = xf[sorted_indices]

X_detrended = np.ones((len(t), 1 + 2*len(dominant_freqs)))  # Intercept + sin/cos terms
X_detrended[:, 0] = 1  # Intercept term

# Add sine and cosine terms for each dominant frequency
for i, freq in enumerate(dominant_freqs):
    X_detrended[:, 2*i+1] = np.sin(2 * np.pi * freq * t)
    X_detrended[:, 2*i+2] = np.cos(2 * np.pi * freq * t)

# Fit regression model on detrended data
model_detrended = LinearRegression()
model_detrended.fit(X_detrended, y)

y_pred = model_detrended.predict(X_detrended)
price_pred = y_pred + drift

def predict(date):
    date_index = (pd.to_datetime(date) - pd.to_datetime(nat_gas_data['Dates'].iloc[0])).days // 30
    if date_index < len(t):
        return price_pred[date_index]
    else:
        future_index = date_index - len(t)
        future_drift_value = a[0] * (len(t) + future_index) + a[1]
        X_future_value = np.ones((1, 1 + 2*len(dominant_freqs)))
        X_future_value[0, 0] = 1
        for i, f in enumerate(dominant_freqs):
            X_future_value[0, 2*i+1] = np.sin(2*np.pi*f*(len(t) + future_index))
            X_future_value[0, 2*i+2] = np.cos(2*np.pi*f*(len(t) + future_index))
        y_future_value = model_detrended.predict(X_future_value)
        return y_future_value[0] + future_drift_value