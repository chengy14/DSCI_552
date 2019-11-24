#
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/27/2019 9:10 AM'

import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, LABELS_NAME
from ay_hw_5.util_data import load_data, train_test_split_by_ratio


def StandardSVC():
	return Pipeline([
		("standardize", StandardScaler()),
		("svc", SVC(kernel="rbf", decision_function_shape='ovr'))
	])


if __name__ == "__main__":
	X_data, Y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	X_train, X_test, Y_train, Y_test = train_test_split_by_ratio(X_data, Y_data, test_size=0.3, random_state=2333)

	SVCParameters = {"gamma": np.logspace(-9, 3, 10), "C": np.logspace(-2, 5, 10)}
	stdSVCParameters = {"svc__gamma": np.logspace(-9, 3, 10), "svc__C": np.logspace(-2, 5, 10)}

	for label_index in range(Y_train.shape[1]):
		y_train = Y_train[:, label_index]
		y_test = Y_test[:, label_index]

		svc = SVC(kernel="rbf", decision_function_shape='ovr')
		stdSVC = StandardSVC()
		gridCV = GridSearchCV(svc, SVCParameters, cv=5, n_jobs=-1)
		stdGridCV = GridSearchCV(stdSVC, stdSVCParameters, cv=5, n_jobs=-1)

		gridCV.fit(X_train, y_train)
		stdGridCV.fit(X_train, y_train)

		bestClf = gridCV.best_estimator_
		bestStdClf = stdGridCV.best_estimator_

		y_predict = bestClf.predict(X_test)
		y_std_predict = bestStdClf.predict(StandardScaler().fit(X_test).transform(X_test))

		print("-------------------------" + LABELS_NAME[label_index] + "---------------------------")
		print("Using RawData, The best params are: ", gridCV.best_params_)
		print("Using Standardized Data, The best params are: ", stdGridCV.best_params_)

		print("Using RawData, The accuracy score is: ", accuracy_score(y_test, y_predict))
		print("Using Standardized Data, The accuracy score is: ", accuracy_score(y_test, y_std_predict))

		print("Using RawData, The hamming loss is: ", hamming_loss(y_test, y_predict))
		print("Using Standardized Data, The hamming loss is: ", hamming_loss(y_test, y_std_predict))
