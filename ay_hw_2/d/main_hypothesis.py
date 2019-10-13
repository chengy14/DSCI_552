# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/16/2019 10:41 PM'

from ay_hw_2.util import load_data
from ay_hw_2.linear_regression.LinearRegression import LinearRegression

import statsmodels.api as sm
import numpy as np
import pandas as pd

if __name__ == "__main__":
	X_data, y_data = load_data("../assets/data.csv")
	labels = ['Intercept', 'Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']

	#OLS
	X_data_constant = sm.add_constant(X_data)
	model = sm.OLS(y_data, X_data_constant)
	results = model.fit()
	print(results.summary())
	# Aaron's own Linear Regression
	# linear_regression = LinearRegression().fit(X_data, y_data)
	# summary = linear_regression.summary()
	#
	# R_square_list = [linear_regression.score(X_data, y_data)]
	# for index in range(4):
	# 	X_single_train = X_data[:, index].reshape(-1, 1)
	# 	linear_regression = LinearRegression().fit(X_single_train, y_data)
	# 	R_square_list.append(linear_regression.score(X_single_train, y_data))
	#
	# summary = summary.join(pd.DataFrame({'R-square': np.array(R_square_list)}))
	# summary.index = labels
	# print(summary.to_string())


