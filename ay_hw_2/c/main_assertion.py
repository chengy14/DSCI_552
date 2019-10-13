# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/16/2019 10:41 PM'

import matplotlib.pyplot as plt

from ay_hw_2.util import load_data
from ay_hw_2.linear_regression.LinearRegression import LinearRegression

# from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
	X, y = load_data("../assets/data.csv")

	labels = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']
	fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(32, 32))

	for index in range(4):
		X_data = X[:, index].reshape(-1, 1)

		# Aaron's own Linear Regression
		linear_regression = LinearRegression().fit(X_data, y)
		y_predict = linear_regression.predict(X_data)

		ax[index, 0].scatter(X_data, y, color='#0984e3', alpha=0.2)
		ax[index, 0].plot(X_data, y_predict, color='#ff7675', alpha=0.8)
		ax[index, 0].grid(True)
		ax[index, 0].set_ylabel("Energy Output")
		ax[index, 0].set_xlabel("Figure c.{} - {}".format(index + 1, labels[index]))
		ax[index, 1].remove()

	plt.show()
