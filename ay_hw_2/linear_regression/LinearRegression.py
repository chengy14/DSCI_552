# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/11/2019 9:19 AM'

import numpy as np
import pandas as pd
from scipy import stats


class LinearRegression:
	'''y = β0 + β1X1 + β2X2 + ... + βnXn'''
	'''intercept = β0'''

	def __init__(self):
		"""Initialize Linear Regression Model"""
		self.coefficient_ = None
		self.intercept_ = None
		self._beta = None

		# Argumented Matrix
		self._X_b_train = None
		self._X_b_test = None

		# Only used for summary()
		self._X_train = None
		self._y_train = None

	def fit(self, X_train, y_train):
		"""Training model through X_train, y_train"""
		assert X_train.shape[0] == y_train.shape[0], \
			"the size of X_train must be equal to the size of y_train"
		self._X_train = X_train
		self._y_train = y_train

		# X_b = [1| X] matrix
		# [beta] = a vector containing (β0, β1, β2, β3, ...βn) shape(-1, 1)
		self._X_b_train = np.hstack([np.ones((len(X_train), 1)), X_train])
		# y_hat (y_predict) = X_b .dot [beta]
		self._beta = np.linalg.inv(self._X_b_train.T.dot(self._X_b_train)).dot(self._X_b_train.T).dot(y_train)
		self.intercept_ = self._beta[0]
		self.coefficient_ = self._beta[1:]

		return self

	def fit_poly(self, X_train, y_train):
		"""Training model through X_train, y_train"""
		assert X_train.shape[0] == y_train.shape[0], \
			"the size of X_train must be equal to the size of y_train"
		self._X_train = X_train
		self._y_train = y_train

		# [1, a, b, a^2, ab, b^2].
		# [beta] = a vector containing (β0, β1, β2, β3, ...βn) shape(-1, 1)
		# y_hat (y_predict) = X_train .dot [beta]
		self._beta = np.linalg.inv(self._X_train.T.dot(self._X_train)).dot(self._X_train.T).dot(y_train)
		self.intercept_ = self._beta[0]
		self.coefficient_ = self._beta[1:]
		self.coefficient_ = np.insert(self.coefficient_, 0, 0)

		return self

	def predict(self, X_predict):
		"""Make prediction based on X_predict Matrix, return a vector with one dimension"""
		assert self.intercept_ is not None and self.coefficient_ is not None, \
			"must fit before predict!"
		assert X_predict.shape[1] == len(self.coefficient_), \
			"the feature number of X_predict must be equal to X_train"

		self._X_b_test = np.hstack([np.ones((len(X_predict), 1)), X_predict])
		return self._X_b_test.dot(self._beta)

	def predict_poly(self, X_predict):
		"""Make prediction based on X_predict Matrix, return a vector with one dimension"""
		assert self.intercept_ is not None and self.coefficient_ is not None, \
			"must fit before predict!"
		assert X_predict.shape[1] == len(self.coefficient_), \
			"the feature number of X_predict must be equal to X_train"

		self._X_b_test = X_predict
		return self._X_b_test.dot(self._beta)

	def get_MSE(self, X_test, y_test):
		y_predict = self.predict(X_test)
		return np.sum((y_test - y_predict) ** 2) / len(y_test)

	def MSE(self):
		'''Equal to the RSS (resdiual sum of square)'''
		if self._X_b_test is None:
			self._X_b_test = np.hstack([np.ones((len(self._X_train), 1)), self._X_train])
		y_predict = self._X_b_test.dot(self._beta)
		return np.sum((self._y_train - y_predict) ** 2) / len(self._y_train)

	def RSS(self):
		'''Equal to the MSE (mean square of error)'''
		if self._X_b_test is None:
			self._X_b_test = np.hstack([np.ones((len(self._X_train), 1)), self._X_train])
		y_predict = self._X_b_test.dot(self._beta)
		return np.sum((self._y_train - y_predict) ** 2) / len(self._y_train)

	def score(self, X_test, y_test):
		'''based on R square and its forumla is 1 - RSS / TSS'''
		y_predict = self.predict(X_test)
		return 1 - np.sum((y_test - y_predict) ** 2) / len(y_test) / np.var(y_test)

	def _standard_error(self):
		if self._X_b_test is None:
			self._X_b_test = np.hstack([np.ones((len(self._X_train), 1)), self._X_train])
		return np.sqrt(self.RSS() * (np.linalg.inv(np.dot(self._X_b_test.T, self._X_b_test)).diagonal()))

	def _t_statistic(self):
		'''t_statistic = beta_hat - beta / std_error'''
		return self._beta / self._standard_error()

	def _p_value(self):
		_t_statistic_list = self._t_statistic()
		return [2 * (1 - stats.t.cdf(np.abs(i), (len(self._X_b_test) - 1))) for i in _t_statistic_list]

	def summary(self, labels=None):
		if labels is not None: labels.insert(0, 'Intercept')
		return pd.DataFrame({
			"Coefficients:": self._beta,
			"Std. Error": self._standard_error(),
			"t-statistic": self._t_statistic(),
			"p-value": np.around(self._p_value(), 5)
		}, index=labels)

	def __repr__(self):
		return "LinearRegression()"
