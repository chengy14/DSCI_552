# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 10:37 AM'

import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

'''Write code for k-nearest neighbors with Minkowski Distance metric '''


class KNeighborsClassifier:

	def __init__(self, n_neighbors, p=2):
		assert n_neighbors >= 1, "n_neighbors must be a valid number"
		self.n_neighbors = n_neighbors
		self.p = p
		self._X_train = None
		self._y_train = None

	def fit(self, X_train, y_train):
		assert X_train.shape[0] == y_train.shape[0], ""
		assert self.n_neighbors <= X_train.shape[0], ""
		self._y_train = y_train
		self._X_train = X_train
		return self

	def predict(self, X_test):
		assert self._X_train is not None and self._y_train is not None, ""
		assert X_test.shape[1] == self._X_train.shape[1], ""

		y_predict = [self._predict(x) for x in X_test]
		return np.array(y_predict)

	def _predict(self, x):
		assert x.shape[0] == self._X_train.shape[1], ""
		# Minkowski Distance metric ==> sum(|x - y|^p)^(1/p)
		distances = [np.sum((abs(item - x) ** self.p)) ** (1 / self.p) for item in self._X_train]
		nearest = np.argsort(distances)
		topK_y = [self._y_train[i] for i in nearest[:self.n_neighbors]]
		votes = Counter(topK_y)

		return votes.most_common(1)[0][0]

	def score(self, X_test, y_test):
		y_predict = self.predict(X_test)
		return accuracy_score(y_test, y_predict)

	def __repr__(self):
		return "KNN(n_neighbors = %d)" % self.n_neighbors
