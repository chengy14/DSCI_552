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
# the idea of this method plotting confusion matrix comes from
# https://stackoverflow.com/questions/39033880/plot-confusion-matrix-sklearn-with-multiple-labels
# I simplify that method a little bit and add some like precision. recall features on it
def plot_confusion_matrix(y_true, y_pred, target_names, title=None):
	confusionMatrix = confusion_matrix(y_true, y_pred)
	precision, recall, _, _ = score(y_true, y_pred, average='macro')

	plt.figure(figsize=(8, 6))
	plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=88)
		plt.yticks(tick_marks, target_names)

	thresh = confusionMatrix.max() / 2
	for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
		plt.text(j, i, "{:,}".format(confusionMatrix[i, j]),
				 horizontalalignment="center",
				 color="white" if confusionMatrix[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label\nOverall Precision={:0.4f}; '
			   'Overall Recall={:0.4f}'.format(precision, recall))
	plt.title(title)
	plt.show()


def plot_roc_curve(y_test, y_score, target_names, title):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(len(target_names)):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	colors = itertools.cycle(['blue', 'red', 'green'])
	for i, color in zip(range(len(target_names)), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=1,
				 label='ROC curve of class {0} (area = {1:0.2f})'
					   ''.format(i, roc_auc[i]))
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

	parameters = {"svc__tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], "svc__C": np.logspace(-2, 5, 10)}
	for label_index in range(Y_train.shape[1]):
		y_train = Y_train[:, label_index]
		y_test = Y_test[:, label_index]

		smote = SMOTE(random_state=2333)
		X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)
		X_test_smote, y_test_smote = smote.fit_sample(X_test, y_test)

		ls_scv = linearStdSVC()
		gridCV = GridSearchCV(ls_scv, parameters, cv=2)
		gridCV.fit(X_train_smote, y_train_smote)

		bestClf = gridCV.best_estimator_
		y_predict = bestClf.predict(X_test_smote)

		plot_confusion_matrix(y_test_smote, y_predict, target_names=Y_LABEL[label_index],
							  title='Confusion Matrix about {} Label'.format(LABELS_NAME[label_index]))
