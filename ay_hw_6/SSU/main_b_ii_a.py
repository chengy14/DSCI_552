#
from warnings import simplefilter

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/5/2019 8:23 PM'

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn.svm import LinearSVC

from ay_hw_6._global import ROOT_PATH, WDBC_FILE_PATH, SPLASH
from ay_hw_6.util_data import load_data, train_test_split_by_class_and_ratio

if __name__ == "__main__":
	simplefilter(action='ignore', category=ConvergenceWarning)
	X_data, y_data = load_data(ROOT_PATH + SPLASH + WDBC_FILE_PATH)
	X_data = Normalizer().fit(X_data).transform(X_data)
	labelEncoder = LabelEncoder().fit(['B', 'M'])
	y_data = labelEncoder.transform(y_data)
	X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																		   random_state=2333, pos_class=0, neg_class=1)

	labeled_X, unlabeled_X, labeled_y, expert = train_test_split_by_class_and_ratio(X_train, y_train, test_size=0.5,
																					random_state=666, pos_class=0,
																					neg_class=1)
	parameters = {"C": np.linspace(0.01, 2, 10)}

	gridCV = GridSearchCV(LinearSVC(penalty="l1", dual=False, max_iter=10000), parameters, cv=5, n_jobs=-1)
	gridCV.fit(labeled_X, labeled_y)

	print("The test score  is:", gridCV.score(X_test, y_test))

	print("The best penalty parameter C is: ", gridCV.best_params_)
