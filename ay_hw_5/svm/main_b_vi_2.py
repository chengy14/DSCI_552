#
from sklearn.multiclass import OneVsRestClassifier

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/27/2019 9:36 AM'

import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC

from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, Y_LABEL, LABELS_NAME
from ay_hw_5.util_data import load_data, train_test_split_by_ratio

warnings.filterwarnings("ignore")


def linearStdSVC():
	return Pipeline([
		("standardize", StandardScaler()),
		("svc", LinearSVC(penalty="l1", multi_class='ovr', dual=False))
	])


#  Citiation
# ----------------------------------------------------------
# the idea of this method plotting multi-class roc curve comes from
# https://stackoverflow.com/questions/33547965/computing-auc-and-roc-curve-from-multi-class-data-in-scikit-learn-sklearn
def plot_roc_curve(y_test, y_score, target_names, title):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(len(target_names)):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	colors = ['#ffa502', '#16a085', '#ff4757', '#747d8c',
			  '#5352ed', '#2ed573', '#c0392b', '#8e44ad']
	for i, color in zip(range(len(target_names)), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=1,
				 label='ROC of class {0} (area = {1:0.2f})'
					   ''.format(target_names[i], roc_auc[i]))
	plt.plot([0, 1], [0, 1], 'k--', lw=1)
	plt.xlim([-0.05, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.show()


if __name__ == "__main__":
	X_data, Y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	X_train, X_test, Y_train, Y_test = train_test_split_by_ratio(X_data, Y_data, test_size=0.3, random_state=2333)

	Cs = [2.1544, 77.4263, 77.4263]
	tols = [1e-6, 0.0001, 0.0001]
	for label_index in range(Y_train.shape[1]):
		y_train = Y_train[:, label_index]
		y_test = Y_test[:, label_index]

		y_train = label_binarize(y_train, classes=Y_LABEL[label_index])
		y_test = label_binarize(y_test, classes=Y_LABEL[label_index])

		classifier = OneVsRestClassifier(
			LinearSVC(penalty="l1", multi_class='ovr', dual=False, tol=tols[label_index], C=Cs[label_index]))
		y_score = classifier.fit(X_train, y_train).decision_function(X_test)

		plot_roc_curve(y_test, y_score, target_names=Y_LABEL[label_index],
					   title='ROC &AUC for Multi-Class about {} Label'
					   .format(LABELS_NAME[label_index]))
