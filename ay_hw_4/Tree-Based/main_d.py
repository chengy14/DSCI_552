#
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/12/2019 5:37 PM'

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import roc_curve, auc

from ay_hw_4._global import ROOT_PATH, APS_TRAIN, APS_TEST, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data, to_binary_numeric

if __name__ == "__main__":
	warnings.simplefilter(action='ignore', category=DataConversionWarning)
	X_train, y_train = load_data(ROOT_PATH + APS_TRAIN, skip_first_row=21, y_column_index=0,
								 assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
								 dropOrNot=True)

	X_test, y_test = load_data(ROOT_PATH + APS_TEST, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
							   dropOrNot=True)

	y_test = to_binary_numeric(y_test, classNeg="neg")
	y_train = to_binary_numeric(y_train, classNeg="neg")

	smote = SMOTE(random_state=2333)
	smote_train_fit = smote.fit_sample(X_train, y_train)
	smote_test_fit = smote.fit_sample(X_test, y_test)
	X_train_smote = pd.DataFrame(smote_train_fit[0])
	y_train_smote = pd.DataFrame(smote_train_fit[1], columns=['class'])
	X_test_smote = pd.DataFrame(smote_test_fit[0])
	y_test_smote = pd.DataFrame(smote_test_fit[1], columns=['class'])

	print("-----------\"After Using SMOTE: (Train)\"-------------")
	print(y_train_smote['class'].value_counts())
	print("-----------\"After Using SMOTE: (Test)\"-------------")
	print(y_test_smote['class'].value_counts())

	randForestClf = RandomForestClassifier(n_estimators=50, random_state=2333, oob_score=True)
	randForestClf.fit(X_train_smote, y_train_smote)
	y_predict = randForestClf.predict(X_test_smote)
	falsePositiveRate, truePositiveRate, thresholds = roc_curve(y_test_smote, y_predict)
	# compute Area Under the Curve (AUC) using the trapezoidal rule
	area = auc(falsePositiveRate, truePositiveRate)

	plt.plot(falsePositiveRate, truePositiveRate, color='red', label='ROC = ' + str(area))
	plt.plot([0, 1], [0, 1], linestyle='dotted')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC & AUC')
	plt.legend()
	plt.show()
	print("-----------\"OOB Score\"-------------")
	print(randForestClf.oob_score_)
