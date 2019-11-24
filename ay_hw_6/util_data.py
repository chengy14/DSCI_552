#
from collections import Counter

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/29/2019 10:14 AM'

import numpy as np
import pandas as pd


def load_data(file_path, X_startIndex=2, X_endIndex=None, y_index=1):
	row_data = pd.read_csv(file_path, encoding="utf-8", header=None)
	if X_endIndex == None:
		X_endIndex = row_data.shape[1]
	X = row_data.iloc[:, X_startIndex:X_endIndex]
	y = row_data.iloc[:, y_index]
	return X.to_numpy(), y.to_numpy()


def train_test_split_by_class_and_ratio(X, y, test_size=0.2, random_state=None, pos_class='B', neg_class='M'):
	if random_state:
		np.random.seed(random_state)
		shuffled_indexes = np.random.permutation(len(X))
		X = X[shuffled_indexes]
		y = y[shuffled_indexes]

	positiveClassAmount = len(X[y == pos_class])
	negativeClassAmount = len(X[y == neg_class])
	positivetestSetLength = round(positiveClassAmount * test_size)
	negativetestSetLength = round(negativeClassAmount * test_size)

	X_test_B = X[y == pos_class][:positivetestSetLength]
	X_test_M = X[y == neg_class][:negativetestSetLength]
	X_test = np.concatenate((X_test_B, X_test_M), axis=0)

	X_train_B = X[y == pos_class][positivetestSetLength:]
	X_train_M = X[y == neg_class][negativetestSetLength:]
	X_train = np.concatenate((X_train_B, X_train_M), axis=0)

	y_test_B = y[y == pos_class][:positivetestSetLength]
	y_test_M = y[y == neg_class][:negativetestSetLength]
	y_test = np.concatenate((y_test_B, y_test_M), axis=0)

	y_train_B = y[y == pos_class][positivetestSetLength:]
	y_train_M = y[y == neg_class][negativetestSetLength:]
	y_train = np.concatenate((y_train_B, y_train_M), axis=0)

	return X_train, X_test, y_train, y_test


def train_test_split_by_exact_number(X, y, test_size=472, random_state=None):
	if random_state:
		np.random.seed(random_state)

	shuffled_indexes = np.random.permutation(len(X))

	test_indexes = shuffled_indexes[:test_size]
	train_indexes = shuffled_indexes[test_size:]

	X_train = X[train_indexes]
	y_train = y[train_indexes]

	X_test = X[test_indexes]
	y_test = y[test_indexes]

	return X_train, X_test, y_train, y_test


def check_y_data(data):
	return True if len(Counter(data).keys()) == 2 else False
