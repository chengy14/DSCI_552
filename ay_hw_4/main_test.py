# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 11:13 PM'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
	x = np.random.uniform(-3, 3, size=100)
	X = x.reshape(-1, 1)

	# assume the relationship between X and Y is linear y = 0.5X + 3 + ε
	y = 3 + 0.5 * x + np.random.normal(0, 0.5, 100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2333)
	print(X_train.shape)
	# Since the formula of MSE is 1 /n * (∑((y_hat - y_true)^2)) and RSS is ∑((y_hat - y_true)^2)
	# so MSE = 1 / n * (RSS), we can use MSE to evaluate our algorithm for convenience

	# get MSE(RSS) fpr the linear regression
	lin_reg = LinearRegression()
	lin_reg.fit(X_train, y_train)
	lin_p1 = lin_reg.predict(X_train)
	lin_p2 = lin_reg.predict(X_test)

	# get MSE(RSS) for the cubic regression
	cubic_reg = Pipeline([("polynomial_features", PolynomialFeatures(degree=3)),
						  ("linear_regression", LinearRegression())])
	cubic_reg.fit(X_train, y_train)
	cub_p1 = cubic_reg.predict(X_train)
	cub_p2 = cubic_reg.predict(X_test)

	# plot figures
	plt.scatter(x, y)
	plt.plot(X_train.tolist(), lin_p1, color='r', label="linear Train")
	plt.scatter(X_train.tolist(), cub_p1.tolist(), color='g', label="cubic Train")
	# plt.plot(X_test.tolist(), y_p1, color='r', label="linear Test")
	# plt.plot(X_test.tolist(), y_p2, color='g', label="cubic Test")
	plt.legend()
	print("MSE for linear regression: ", mean_squared_error(y_train, lin_p1))
	print("MSE for cubic regression: ", mean_squared_error(y_train, cub_p1))
	print("------------------")
	print("MSE for linear regression: ", mean_squared_error(y_test, lin_p2))
	print("MSE for cubic regression: ", mean_squared_error(y_test, cub_p2))
	plt.show()
