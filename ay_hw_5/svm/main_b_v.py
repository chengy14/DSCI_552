#
from sklearn.model_selection import GridSearchCV

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/27/2019 9:36 AM'

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ay_hw_5.util_data import load_data, train_test_split_by_ratio
from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, LABELS_NAME, Y_LABEL, MODEL_NAMES
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def linearStdSVC():
	return Pipeline([
		("standardize", StandardScaler()),
		("svc", LinearSVC(penalty="l1", multi_class='ovr', dual=False))
	])


def plot_grouped_bar(y_values, x_labels, legend_labels):
	colors = ['#EE5A24', '#009432', '#0652DD']
	x_pos = np.arange(len(x_labels))
	width = 0.25
	fig, ax = plt.subplots(figsize=(7, 4))
	for index in range(3):
		ax.bar(x_pos - width * index, y_values[index], alpha=0.5, color=colors[index], width=width,
			   label=legend_labels[index])

	ax.grid(True)
	ax.set_title('Classifier Chain Ensemble Performance Comparison')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(x_labels, rotation='vertical')
	ax.set_ylabel('Jaccard Similarity Score')
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	X_data, Y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	X_train, X_test, Y_train, Y_test = train_test_split_by_ratio(X_data, Y_data, test_size=0.3, random_state=2333)

	parameters = {"svc__tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], "svc__C": np.logspace(-2, 5, 10)}
	X_test = StandardScaler().fit(X_test).transform(X_test)

	model_scores = []

	sss = 9

	for label_index in range(Y_train.shape[1]):
		y_train = Y_train[:, label_index]
		y_test = Y_test[:, label_index]
		#
		labelEncoder = LabelEncoder().fit(Y_LABEL[label_index])
		y_train = labelEncoder.transform(y_train)
		y_test = labelEncoder.transform(y_test)
		#
		# ls_scv = linearStdSVC()
		# gridCV = GridSearchCV(ls_scv, parameters, cv=5)
		# gridCV.fit(X_train, y_train)
		chains = [ClassifierChain(SVC(kernel='rbf'), order='random', random_state=i) for i in range(10)]
		for classifier in chains:
			classifier.fit(X_train, y_train.reshape(-1, 1))


		#  Citiation
		# ----------------------------------------------------------
		# the following idea about using jaccard_score is come from
		# https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html
		# but I choose different way to average the score and plot it in my way
		y_pred_chains = np.array([classifier.predict(X_test) for classifier in chains])

		chain_jaccard_scores = [jaccard_score(y_test.reshape(-1, 1), y_pred_chain >= .5, average='micro') for
								y_pred_chain in y_pred_chains]

		y_pred_ensemble = y_pred_chains.mean(axis=0)
		ensemble_jaccard_score = jaccard_score(y_test, y_pred_ensemble >= .5, average='micro')

		chain_jaccard_scores.append(ensemble_jaccard_score)
		model_scores.append(chain_jaccard_scores)

	plot_grouped_bar(model_scores, MODEL_NAMES, LABELS_NAME)
