import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

import warnings
warnings.filterwarnings('ignore')

# Load the datasets
train = pd.read_csv(r'C:\Users\SPARKZ EDUCATION\Desktop\My_Projects\sales_pred\train.csv')
test = pd.read_csv(r'C:\Users\SPARKZ EDUCATION\Desktop\My_Projects\sales_pred\test.csv')

print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)
print(train.head())

# Fill missing values for 'Item_Weight' and 'Outlet_Size'
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True)

# Add dummy 'seasonality', 'promotions', and 'economic_indicators' columns (replace with actual data if available)
train['Seasonality'] = np.random.randint(0, 2, train.shape[0])
train['Promotions'] = np.random.randint(0, 2, train.shape[0])
train['Economic_Indicator'] = np.random.random(train.shape[0])

test['Seasonality'] = np.random.randint(0, 2, test.shape[0])
test['Promotions'] = np.random.randint(0, 2, test.shape[0])
test['Economic_Indicator'] = np.random.random(test.shape[0])

# One-hot encode categorical variables
train = pd.get_dummies(train, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], drop_first=True)
test = pd.get_dummies(test, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], drop_first=True)

X = train.drop(['Item_Outlet_Sales', 'Item_Identifier'], axis=1)
y = train['Item_Outlet_Sales']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_val)

#linear regression model
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Linear Regression Performance:")
print(f"RMSE: {rmse}, MAE: {mae}, R-squared: {r2}")

X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=32)

#LSTM
y_pred_lstm = lstm_model.predict(X_val_lstm)

#LSTM model
mse_lstm = mean_squared_error(y_val, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(y_val, y_pred_lstm)
r2_lstm = r2_score(y_val, y_pred_lstm)

print("LSTM Performance:")
print(f"RMSE: {rmse_lstm}, MAE: {mae_lstm}, R-squared: {r2_lstm}")

param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

print("Best Parameters for Linear Regression:", grid.best_params_)

X_test = test.drop(['Item_Identifier'], axis=1)
test['Item_Outlet_Sales'] = grid.predict(X_test)

test[['Item_Identifier', 'Item_Outlet_Sales']].to_csv(r'C:\Users\SPARKZ EDUCATION\Desktop\My_Projects\sales_pred\sample_submission.csv', index=False)
print("Predictions saved to sample_submission.csv")

