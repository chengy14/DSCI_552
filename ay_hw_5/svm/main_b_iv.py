# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/27/2019 9:36 AM'

import warnings

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, LABELS_NAME
from ay_hw_5.util_data import load_data, train_test_split_by_ratio

warnings.filterwarnings("ignore")


def linearStdSVC():
	return Pipeline([
		("standardize", StandardScaler()),
		("svc", LinearSVC(penalty="l1", multi_class='ovr', dual=False))
	])


if __name__ == "__main__":
	X_data, Y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	X_train, X_test, Y_train, Y_test = train_test_split_by_ratio(X_data, Y_data, test_size=0.3, random_state=2333)

	parameters = {"svc__tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], "svc__C":  np.logspace(-2, 5, 10)}

	for label_index in range(Y_train.shape[1]):
		y_train = Y_train[:, label_index]
		y_test = Y_test[:, label_index]

		smote = SMOTE(random_state=2333)
		X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)
		X_test_smote, y_test_smote = smote.fit_sample(X_test, y_test)

		ls_scv = linearStdSVC()
		gridCV = GridSearchCV(ls_scv, parameters, cv=10)
		gridCV.fit(X_train_smote, y_train_smote)

		bestClf = gridCV.best_estimator_
		y_predict = bestClf.predict(X_test_smote)

		print("-------------------------" + LABELS_NAME[label_index] + "---------------------------")
		print("Using RawData, The best params are: ", gridCV.best_params_)

		print("Using RawData, The accuracy score is: ", accuracy_score(y_test_smote, y_predict))

		print("Using RawData, The hamming loss is: ", hamming_loss(y_test_smote, y_predict))
