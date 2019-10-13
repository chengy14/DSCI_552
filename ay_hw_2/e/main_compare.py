# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/18/2019 9:04 PM'

from ay_hw_2.util import load_data, train_test_split_by_ratio
from ay_hw_2.linear_regression.LinearRegression import LinearRegression

import matplotlib.pyplot as plt

if __name__ == "__main__":
	X_data, y_data = load_data("../assets/data.csv")
	colors = ['#f6b93b', '#1e3799', '#079992', '#eb2f06']
	labels = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']
	# Aaron's own Linear Regression
	linear_regression = LinearRegression().fit(X_data, y_data)
	mult_reg_coef = linear_regression.coefficient_

	single_reg_coef = list()
	for index in range(4):
		X_single_train = X_data[:, index].reshape(-1, 1)
		linear_regression = LinearRegression().fit(X_single_train, y_data)
		single_reg_coef.append(linear_regression.coefficient_[0])

	for index in range(0, 4):
		plt.scatter(single_reg_coef[index], mult_reg_coef[index],
					label=labels[index], color=colors[index], alpha=0.7)
	plt.vlines(single_reg_coef, -2.1, 0.5, colors='gray', linestyles='dotted')
	plt.hlines(mult_reg_coef, -3, 1.8, colors='gray', linestyles='dotted')
	plt.xlabel("univariate regression coefficients")
	plt.ylabel("multiple regression coefficient")
	plt.grid(True)
	plt.legend()
	plt.show()
