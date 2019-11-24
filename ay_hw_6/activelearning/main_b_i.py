#
import warnings

from sklearn.metrics import accuracy_score
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
	passiveOverallErrorList = list()
	for i in tqdm(range(2)):
		X_train, X_test, y_train, y_test = train_test_split_by_exact_number(X_data, y_data, test_size=472,
																			random_state=i)
		indexListLength = len(X_train)
		indexList = np.arange(0, indexListLength)
		X_train_pool = np.array([[]])
		y_train_pool = np.array([])
		tempErrorList = list()
		for __ in tqdm(range(90)):
			indexes = random.sample(indexList.tolist(), 10)
			while not check_y_data(y_train[indexes]):
				indexes = random.sample(indexList.tolist(), 10)
			indexList = np.setdiff1d(indexList, np.array(indexes))
			X_train_pool = np.append(X_train_pool, X_train[indexes]).reshape(-1, 4)
			y_train_pool = np.append(y_train_pool, y_train[indexes]).reshape(-1, )
			ls_scv = linearStdSVC()
			cv = 10 if len(y_train_pool) > 10 else 5
			gridCV = GridSearchCV(ls_scv, parameters, cv=cv, n_jobs=-1, scoring='accuracy')
			gridCV.fit(X_train_pool, y_train_pool)
			tempErrorList.append(1 - gridCV.score(X_test, y_test))

		passiveOverallErrorList.append(tempErrorList)

	plot_mult_sets(np.arange(10, 901, 10).tolist(), passiveOverallErrorList, title="Passive Learning")
