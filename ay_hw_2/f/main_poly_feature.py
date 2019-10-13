# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/18/2019 9:39 PM'

import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from ay_hw_2.util import load_data
from ay_hw_2.linear_regression.LinearRegression import LinearRegression

if __name__ == "__main__":
	X_data, y_data = load_data("../assets/data.csv")
	labels = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']

	poly = PolynomialFeatures(degree=3)
	fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 16))

	single_reg_coef = list()
	for index in range(4):
		X_train = X_data[:, index].reshape(-1, 1)
		X_train_poly = poly.fit_transform(X_train)

		linear_regression = LinearRegression().fit(X_train, y_data)
		y_predict = linear_regression.predict(X_train)

		linear_regression_poly = LinearRegression().fit_poly(X_train_poly, y_data)
		y_predict_poly = linear_regression_poly.predict_poly(X_train_poly)
		print("-----------------------{}------------------".format(labels[index]))
		print(linear_regression_poly.summary(['β1', 'β2', 'β3']))

		ax[index, 0].scatter(X_train, y_data, color='#0984e3', label='Data Points', alpha=0.1)
		ax[index, 0].plot(X_train, y_predict, color='#2d3436', label='Linear', alpha=0.8)
		ax[index, 0].scatter(X_train, y_predict_poly, color='#ff7f50', s=10, label='Degree 3', alpha=0.8)
		ax[index, 0].set_ylabel("Energy Output")
		ax[index, 0].set_xlabel("Figure f.{} - {}".format(index + 1, labels[index]))
		ax[index, 0].grid(True)
		ax[index, 0].legend()
		ax[index, 1].remove()

	fig.tight_layout()
	plt.show()
