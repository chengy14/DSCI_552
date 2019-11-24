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

from ay_hw_4._global import ROOT_PATH, APS_SHRINK, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data, to_binary_numeric

GENERATED_SMOTE_TRAIN_DATA_FILE_PATH = './gen_smote_train_shrink_data_set.csv'

if __name__ == "__main__":
	X_train, y_train = load_data(ROOT_PATH + APS_SHRINK, skip_first_row=21, y_column_index=0,
								 assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
								 dropOrNot=True)

	smote = SMOTE(random_state=2333)
	smote_train_fit = smote.fit_sample(X_train, y_train)
	X_train_smote = pd.DataFrame(smote_train_fit[0])
	y_train_smote = pd.DataFrame(smote_train_fit[1], columns=['class'])
	export_smote_train_data = pd.concat([y_train_smote, X_train_smote], axis=1)

	# export data to csv
	export_smote_train_data.to_csv(GENERATED_SMOTE_TRAIN_DATA_FILE_PATH, sep=',', index=False)
	smote_train_data = convert.load_any_file(filename=GENERATED_SMOTE_TRAIN_DATA_FILE_PATH)
	smote_train_data.class_is_first()

	# load logistic model tree algorithm
	log_tree = Classifier(classname="weka.classifiers.trees.LMT")
	eval_smote_train_obj = Evaluation(smote_train_data)
	eval_smote_train_obj.crossvalidate_model(classifier=log_tree, data=smote_train_data, num_folds=5, rnd=Random(1))
	print("SMOTE Train CV (5-folds) Error = %.2f%%" % (eval_smote_train_obj.percent_incorrect))
	print(eval_smote_train_obj.matrix())
	print("=================\"Summary\"====================")
	print(eval_smote_train_obj.summary())

	log_tree.build_classifier(smote_train_data)
	y_predict = eval_smote_train_obj.test_model(log_tree, smote_train_data)

	y_train_smote = to_binary_numeric(y_train_smote, classNeg="neg")

	falsePositiveRate, truePositiveRate, thresholds = roc_curve(y_train_smote, y_predict, pos_label=0)
	# compute Area Under the Curve (AUC) using the trapezoidal rule
	area = auc(falsePositiveRate, truePositiveRate)

	plt.plot(falsePositiveRate, truePositiveRate, color='red', label='ROC = ' + str(area))
	plt.plot([0, 1], [0, 1], linestyle='dotted')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC & AUC (SMOTE Train)')
	plt.legend()
	plt.show()
