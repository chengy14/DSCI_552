#
from imblearn.over_sampling import SMOTE

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/12/2019 6:11 PM'

import weka.core.jvm as jvm

jvm.start()
import weka.core.converters as convert
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

from ay_hw_4._global import ROOT_PATH, APS_TEST, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data, to_binary_numeric

GENERATED_SMOTE_TEST_DATA_FILE_PATH = './gen_smote_test_data_set.csv'

if __name__ == "__main__":
	X_test, y_test = load_data(ROOT_PATH + APS_TEST, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
							   dropOrNot=True)

	smote = SMOTE(random_state=2333)
	smote_test_fit = smote.fit_sample(X_test, y_test)
	X_test_smote = pd.DataFrame(smote_test_fit[0])
	y_test_smote = pd.DataFrame(smote_test_fit[1], columns=['class'])
	export_smote_test_data = pd.concat([y_test_smote.head(500), X_test_smote.head(500)], axis=1)

	# export data to csv
	export_smote_test_data.to_csv(GENERATED_SMOTE_TEST_DATA_FILE_PATH, sep=',', index=False)
	smote_test_data = convert.load_any_file(filename=GENERATED_SMOTE_TEST_DATA_FILE_PATH)
	smote_test_data.class_is_first()

	# load logistic model tree algorithm
	log_tree = Classifier(classname="weka.classifiers.trees.LMT")
	eval_smote_test_obj = Evaluation(smote_test_data)
	eval_smote_test_obj.crossvalidate_model(classifier=log_tree, data=smote_test_data, num_folds=5, rnd=Random(1))
	print("SMOTE Test CV (5-folds) Error = %.2f%%" % (eval_smote_test_obj.percent_incorrect))
	print(eval_smote_test_obj.matrix())
	print("=================\"Summary\"====================")
	print(eval_smote_test_obj.summary())

	log_tree.build_classifier(smote_test_data)
	y_predict = eval_smote_test_obj.test_model(log_tree, smote_test_data)

	y_test = to_binary_numeric(y_test.head(500), classNeg="neg")

	falsePositiveRate, truePositiveRate, thresholds = roc_curve(y_test, y_predict)
	# compute Area Under the Curve (AUC) using the trapezoidal rule
	area = auc(falsePositiveRate, truePositiveRate)

	plt.plot(falsePositiveRate, truePositiveRate, color='red', label='ROC = ' + str(area))
	plt.plot([0, 1], [0, 1], linestyle='dotted')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC & AUC (SMOTE Test)')
	plt.legend()
	plt.show()
