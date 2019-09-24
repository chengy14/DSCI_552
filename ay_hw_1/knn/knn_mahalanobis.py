# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 10:37 AM'

import numpy as np
import scipy as sp
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

'''Write code for k-nearest neighbors with Mahalanobis Distance metric '''


class KNeighborsClassifier:

	def __init__(self, n_neighbors = 3):
		assert n_neighbors >= 1, "n_neighbors must be a valid number"
		self.n_neighbors = n_neighbors
		self._X_train = None
		self._y_train = None

	def mahalanobis(self, X_test, data=None, cov=None):
		"""Compute the Mahalanobis Distance between each row of X_test and the data
		X_test   : vector or matrix of data with, say, p columns.
		data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
		cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
		"""
		X_minus_mu = X_test - np.mean(data)
		if not cov:
			cov = np.cov(data.T)
		inv_covmat = sp.linalg.inv(cov)
		left_term = np.dot(X_minus_mu, inv_covmat)
		mahal = np.dot(left_term, X_minus_mu.T)
		return mahal.diagonal()

	def fit(self, X_train, y_train):
		assert X_train.shape[0] == y_train.shape[0], ""
		assert self.n_neighbors <= X_train.shape[0], ""
		self._y_train = y_train
		self._X_train = X_train
		self._X_train_normal = self._X_train[self._y_train == 0, :]
		self._X_train_abnormal = self._X_train[self._y_train == 1, :]
		return self

	def predict(self, X_test):
		assert self._X_train is not None and self._y_train is not None, ""
		assert X_test.shape[1] == self._X_train.shape[1], ""

		return np.array([np.argmax(row) for row in self.predict_proba(X_test)])

	def predict_proba(self, X_test):
		# assert x.shape[0] == self._X_train.shape[1], ""
		# Mahalanobis Distance metric
		distances = [(normal, abnormal) for normal, abnormal in
						 zip(self.mahalanobis(X_test, self._X_train_normal), self.mahalanobis(X_test, self._X_train_abnormal))]
		return np.array([(1 - n / (p + n), 1 - p / (p + n)) for p, n in distances])

	def score(self, X_test, y_test):
		y_predict = self.predict(X_test)
		return accuracy_score(y_test, y_predict)

	def __repr__(self):
		return "KNN(n_neighbors = %d)" % self.n_neighbors
