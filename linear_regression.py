# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
np.set_printoptions(suppress=True)

data = pd.read_csv('Advertising.csv')
data['Radio'] += 0.0001

data['TV2'] = data['TV'] ** 2
data['TV3'] = data['TV'] ** 3
data['TV_LOG'] = np.log(data['TV'])
data['TV_REV'] = 1/data['TV']

data['Radio2'] = data['Radio'] ** 2
data['Radio3'] = data['Radio'] ** 3
data['Radio_LOG'] = np.log(data['Radio'])
data['Radio_REV'] = 1 / data['Radio']

print('Pearson Corr = \n', data.corr())

x_cols = ['TV', 'Radio', 'TV2', 'TV3', 'TV_LOG', 'TV_REV', 'Radio2', 'Radio3', 'Radio_LOG', 'Radio_REV']
x = data[x_cols]
mms = MinMaxScaler()
x = mms.fit_transform(x)
y = data['Sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_, model.intercept_)
for col_name, coef in zip(x_cols, model.coef_):
    print('\t', col_name, coef)
y_train_pred = model.predict(x_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
print('Train Set MAE = %f, MSE = %f, RMSE = %f' % (mae_train, mse_train, rmse_train))
print(rmse_train / np.mean(y_train))
print(np.mean(np.abs(y_train - y_train_pred) / y_train))

y_test_pred = model.predict(x_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
print('Test Set MAE = %f, MSE = %f, RMSE = %f' % (mae_test, mse_test, rmse_test))
print(rmse_test / np.mean(y_test))
print(np.mean(np.abs(y_test - y_test_pred) / y_test))
