# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/20/2019 2:21 PM'

import numpy as np
from matplotlib import pyplot as plt
from ay_hw_2.util import load_data, train_test_split_by_ratio
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

	X_data, y_data = load_data("../assets/data.csv")
	transformer = Normalizer().fit(X_data)
	X_train_normalize = transformer.transform(X_data)

	alternative_k = np.arange(1, 101)
	train_error_normal = np.empty(len(alternative_k))
	test_error_normal = np.empty(len(alternative_k))
	train_error_raw = np.empty(len(alternative_k))
	test_error_raw = np.empty(len(alternative_k))

	X_train, X_test, y_train, y_test = train_test_split_by_ratio(X_data, y_data, random_state=2333)
	X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split_by_ratio(X_train_normalize, y_data,
																							 random_state=2333)

	for index, k in enumerate(alternative_k):
		knn_normalize = KNeighborsRegressor(n_neighbors=k).fit(X_train_normal, y_train_normal)
		knn_raw = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)

		y_train_normal_predict = knn_normalize.predict(X_train_normal)
		y_test_normal_predict = knn_normalize.predict(X_test_normal)
		y_train_predict = knn_raw.predict(X_train)
		y_test_predict = knn_raw.predict(X_test)

		train_error_normal[index] = mean_squared_error(y_train_normal,y_train_normal_predict)

		test_error_normal[index] = mean_squared_error(y_test_normal, y_test_normal_predict)

		train_error_raw[index] = mean_squared_error(y_train, y_train_predict)

		test_error_raw[index] = mean_squared_error(y_test, y_test_predict)

	lowest_error_test_normal = np.min(test_error_normal)
	lowest_error_test_raw = np.min(test_error_raw)
	best_normal_k_ = np.reshape(np.argwhere(test_error_normal == lowest_error_test_normal), (1, -1))[0] + 1
	best_raw_k_ = np.reshape(np.argwhere(test_error_raw == lowest_error_test_raw), (1, -1))[0] + 1

	plt.title('k-NN Regression Varying number of neighbors')
	plt.plot(1 / alternative_k, test_error_normal,
			 label='Test Error(Normalize) = ' + str(lowest_error_test_normal)[:5])
	plt.plot(1 / alternative_k, train_error_normal, label='Train Error Rate(Normalize)')
	plt.plot(1 / alternative_k, test_error_raw,
			 label='Test Error Rate(Raw) = ' + str(lowest_error_test_raw)[:5])
	plt.plot(1 / alternative_k, train_error_raw, label='Train Error Rate(Raw)')
	plt.vlines(1 / best_normal_k_, 0, 20, colors='gray', label='1 / k = ' + str(1 / best_normal_k_)[:5],
			   linestyles='dotted')
	plt.hlines(lowest_error_test_normal, 0, 1, colors='gray', linestyles='dotted')
	plt.vlines(1 / best_raw_k_, 0, 20, colors='red', label='1 / k = ' + str(1 / best_raw_k_)[:5],
			   linestyles='dotted')
	plt.hlines(lowest_error_test_raw, 0, 1, colors='red', linestyles='dotted')
	plt.legend()
	plt.grid(True)
	plt.xlabel('1 / Number of neighbors')
	plt.ylabel('Mean Squared Error')
	plt.show()
