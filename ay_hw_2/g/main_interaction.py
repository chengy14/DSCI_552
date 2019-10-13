# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/19/2019 8:42 PM'

from ay_hw_2.util import load_data
from ay_hw_2.linear_regression.LinearRegression import LinearRegression
import numpy as np

if __name__ == "__main__":
	X_data, y_data = load_data("../assets/data.csv")
	labels = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity', 'Temperature x Exhaust Vacuum',
			  'Temperature x Ambient Pressure', 'Temperature x Relative Humidity', 'Exhaust Vacuum x Ambient Pressure',
			  'Exhaust Vacuum x Relative Humidity', 'Ambient Pressure x Relative Humidity']

	for start in range(4):
		for end in range(start + 1, 4):
			temp = X_data[:, start].reshape(-1, 1) * X_data[:, end].reshape(-1, 1)
			X_data = np.append(X_data, temp, 1)

	linear_regression = LinearRegression().fit(X_data, y_data)
	print(linear_regression.summary(labels).to_string())
