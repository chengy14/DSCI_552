#
import warnings
from collections import Counter

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/8/2019 10:04 AM'

import numpy as np
import random
from ay_hw_6._global import ROOT_PATH, BANK_NOTE_FILE_PATH, SPLASH
from ay_hw_6.util_data import load_data, train_test_split_by_exact_number, check_y_data
from ay_hw_6.util_plot import plot_mult_sets

warnings.filterwarnings("ignore")


def linearStdSVC():
	return Pipeline([
		("svc", LinearSVC(penalty="l1", dual=False, max_iter=5000))
	])


if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + SPLASH + BANK_NOTE_FILE_PATH, X_startIndex=0, X_endIndex=4, y_index=-1)
	X_data = StandardScaler().fit(X_data).transform(X_data)


	parameters = {"svc__C": np.linspace(1, 5, 10)}
	activeOverallErrorList = list()
	for i in tqdm(range(5)):
		X_train, X_test, y_train, y_test = train_test_split_by_exact_number(X_data, y_data, test_size=472,
																			random_state=i)
		indexListLength = len(X_train)
		indexList = np.arange(0, indexListLength)
		X_train_pool = np.array([[]])
		y_train_pool = np.array([])
		tempErrorList = list()

		##If all selected data points are from one class, select another set of 10 data points randomly
		indexes = random.sample(indexList.tolist(), 10)
		while not check_y_data(y_train[indexes]):
			indexes = random.sample(indexList.tolist(), 10)
		indexList = np.setdiff1d(indexList, np.array(indexes))

		X_train_pool = np.append(X_train_pool, X_train[indexes]).reshape(-1, 4)
		y_train_pool = np.append(y_train_pool, y_train[indexes]).reshape(-1, )
		y_train_pool = y_train_pool.astype('int32')

		ls_scv = linearStdSVC()
		gridCV = GridSearchCV(ls_scv, parameters, cv=5, n_jobs=-1, scoring='accuracy')
		gridCV.fit(X_train_pool, y_train_pool)
		tempErrorList.append(1 - gridCV.score(X_test, y_test))

		for __ in tqdm(range(89)):
			remainingData = X_train[indexList]
			distances = gridCV.best_estimator_.decision_function(remainingData).reshape(-1, 1)
			remainingData = np.hstack((remainingData, abs(distances)))
			closetIndexes = remainingData[:, 4].argsort()[:10]
			firstCloset10XPoints = X_train[closetIndexes]
			firstCloset10YPoints = y_train[closetIndexes]
			indexList = np.setdiff1d(indexList, np.array(closetIndexes))
			X_train_pool = np.append(X_train_pool, firstCloset10XPoints).reshape(-1, 4)
			y_train_pool = np.append(y_train_pool, firstCloset10YPoints).reshape(-1, )

			ls_scv = linearStdSVC()
			gridCV = GridSearchCV(ls_scv, parameters, cv=10, n_jobs=-1, scoring='accuracy')
			gridCV.fit(X_train_pool, y_train_pool)
			tempErrorList.append(1 - gridCV.score(X_test, y_test))

		activeOverallErrorList.append(tempErrorList)

	plot_mult_sets(np.arange(10, 901, 10).tolist(), activeOverallErrorList, title="Active Learning")
