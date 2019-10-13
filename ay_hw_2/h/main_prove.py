# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/19/2019 8:42 PM'

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ay_hw_2.util import load_data, train_test_split_by_ratio
from ay_hw_2.linear_regression.LinearRegression import LinearRegression


def get_all_possible_interaction_data(X_data):
	# orginal_data have 4 columns => A B C D
	# append AB  AC  AD  BC  BD  CD to the orginal_data obj
	for start in range(4):
		for end in range(start + 1, 4):
			temp = X_data[:, start].reshape(-1, 1) * X_data[:, end].reshape(-1, 1)
			X_data = np.append(X_data, temp, 1)
	# append A^2 B^2 C^2 D^2 to the orginal_data obj
	for index in range(4):
		temp = X_data[:, index].reshape(-1, 1) * X_data[:, index].reshape(-1, 1)
		X_data = np.append(X_data, temp, axis=1)

	return X_data


if __name__ == "__main__":
	X_data, y_data = load_data("../assets/data.csv")
	X_train, X_test, y_train, y_test = train_test_split_by_ratio(X_data, y_data, test_size=0.3, random_state=2333)
	linear_regression = LinearRegression().fit(X_train, y_train)
	print("Before Linear Regression Training MSE:", linear_regression.get_MSE(X_train, y_train))
	print("Before Linear Regression Testing MSE:", linear_regression.get_MSE(X_test, y_test))

	print("----------------------------------------------------------------")
	labels = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']
	combination_labels = ['Temperature x Exhaust Vacuum', 'Temperature x Ambient Pressure',
						  'Temperature x Relative Humidity', 'Exhaust Vacuum x Ambient Pressure',
						  'Exhaust Vacuum x Relative Humidity', 'Ambient Pressure x Relative Humidity',
						  'Temperature^2', 'Exhaust Vacuum^2', 'Ambient Pressure^2', 'Relative Humidity^2']

	X_data = get_all_possible_interaction_data(X_data)
	X_train, X_test, y_train, y_test = train_test_split_by_ratio(X_data, y_data, test_size=0.3, random_state=2333)
	X_train = pd.DataFrame(X_train, columns=labels + combination_labels)
	OLS_linear = sm.OLS(y_train, X_train)
	OLS_linear_results = OLS_linear.fit()
	print(OLS_linear_results.summary())
	remaining_var = list(set([key for key, p_value in OLS_linear_results.pvalues.items() if p_value <= 0.05]) - set(labels))
	print("----------------------------------------------------------------")
	print("The remaining significant variables: ", remaining_var)

	# ['Exhaust Vacuum x Ambient Pressure', 'Ambient Pressure^2', 'Exhaust Vacuum^2',
	# 'Ambient Pressure x Relative Humidity', 'Temperature x Ambient Pressure', 'Temperature x Exhaust Vacuum']
	X_data = pd.DataFrame(X_data, columns=labels+combination_labels)
	X_data = X_data[remaining_var].to_numpy()
	X_train, X_test, y_train, y_test = train_test_split_by_ratio(X_data, y_data, test_size=0.3, random_state=2333)
	linear_regression = LinearRegression().fit(X_train, y_train)
	print("After Linear Regression Training MSE:", linear_regression.get_MSE(X_train, y_train))
	print("After Linear Regression Testing MSE:", linear_regression.get_MSE(X_test, y_test))
#

