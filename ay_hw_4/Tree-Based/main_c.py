# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/9/2019 11:11 PM'

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from ay_hw_4._global import ROOT_PATH, APS_TRAIN, APS_TEST, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data, to_binary_numeric

if __name__ == "__main__":
	X_train, y_train = load_data(ROOT_PATH + APS_TRAIN, skip_first_row=21, y_column_index=0,
								 assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
								 dropOrNot=True)

	X_test, y_test = load_data(ROOT_PATH + APS_TEST, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
							   dropOrNot=True)

	y_train = to_binary_numeric(y_train, classNeg="neg")
	y_test = to_binary_numeric(y_test, classNeg="neg")

	randForestClf = RandomForestClassifier(n_estimators=50, random_state=2333, oob_score=True)
	randForestClf.fit(X_train, y_train)
	y_predict = randForestClf.predict(X_test)
	falsePositiveRate, truePositiveRate, thresholds = roc_curve(y_test, y_predict)
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
