#

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/29/2019 10:14 AM'

import numpy as np


def load_data(file_path):
	"""read records from csv file"""
	row_data = np.genfromtxt(file_path, dtype=None, delimiter=',', encoding='utf-8')[1:]
	X = np.array(row_data[:, :-4], dtype=float)
	y = np.array(row_data[:, row_data.shape[1] - 4: -1], dtype=str)
	return X, y


def train_test_split_by_ratio(X, y, test_size=0.3, random_state=None):
	"""According to the test_size, split the row data into X_train, X_test, y_train, y_test"""
	assert X.shape[0] == y.shape[0], \
		"the size of X must be equal to the size of y"
	assert 0.0 <= test_size <= 1.0, \
		"test_size must be valid"

	if random_state:
		np.random.seed(random_state)

	shuffled_indexes = np.random.permutation(len(X))

	test_size = int(len(X) * test_size)
	test_indexes = shuffled_indexes[:test_size]
	train_indexes = shuffled_indexes[test_size:]

	X_train = X[train_indexes]
	y_train = y[train_indexes]

	X_test = X[test_indexes]
	y_test = y[test_indexes]

	return X_train, X_test, y_train, y_test
