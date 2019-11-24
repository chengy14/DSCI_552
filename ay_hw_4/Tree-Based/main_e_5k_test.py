#

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

GENERATED_TEST_DATA_FILE_PATH = './gen_test_data_set.csv'

if __name__ == "__main__":
	X_test, y_test = load_data(ROOT_PATH + APS_TEST, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
							   dropOrNot=True)
	export_test_data = pd.concat([y_test.head(500), X_test.head(500)], axis=1)

	# export data to csv
	export_test_data.to_csv(GENERATED_TEST_DATA_FILE_PATH, sep=',', index=False)
	test_data = convert.load_any_file(filename=GENERATED_TEST_DATA_FILE_PATH)
	test_data.class_is_first()

	# load logistic model tree algorithm
	log_tree = Classifier(classname="weka.classifiers.trees.LMT")
	eval_test_obj = Evaluation(test_data)
	eval_test_obj.crossvalidate_model(classifier=log_tree, data=test_data, num_folds=5, rnd=Random(1))
	print("Test CV (10-folds) Error = %.2f%%" % (eval_test_obj.percent_incorrect))
	print(eval_test_obj.matrix())
	print("=================\"Summary\"====================")
	print(eval_test_obj.summary())

	log_tree.build_classifier(test_data)
	y_predict = eval_test_obj.test_model(log_tree, test_data)

	y_test = to_binary_numeric(y_test.head(500), classNeg="neg")

	falsePositiveRate, truePositiveRate, thresholds = roc_curve(y_test, y_predict, pos_label=0)
	# compute Area Under the Curve (AUC) using the trapezoidal rule
	area = auc(falsePositiveRate, truePositiveRate)

	plt.plot(falsePositiveRate, truePositiveRate, color='red', label='ROC = ' + str(area))
	plt.plot([0, 1], [0, 1], linestyle='dotted')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC & AUC (Test)')
	plt.legend()
	plt.show()
