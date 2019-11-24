#

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/27/2019 9:36 AM'

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, MODEL_NAMES
from ay_hw_5.util_data import load_data, train_test_split_by_ratio

warnings.filterwarnings("ignore")


def linearStdSVC():
	return Pipeline([
		("standardize", StandardScaler()),
		("svc", LinearSVC(penalty="l1", multi_class='ovr', dual=False))
	])


def encodeYData(Y_data):
	x = Y_data.shape[0]
	y = Y_data.shape[1]
	flattenedData = np.array(Y_data.reshape(-1, ).tolist())
	mapResult = pd.factorize(flattenedData)
	return np.array(mapResult[0]).reshape(x, y)


if __name__ == "__main__":
	X_data, Y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	Y_data = encodeYData(Y_data)
	X_train, X_test, Y_train, Y_test = train_test_split_by_ratio(X_data, Y_data,
																 test_size=0.3, random_state=2333)

	chains = [ClassifierChain(linearStdSVC(), order='random', random_state=i) for i in range(10)]
	for classifier in chains:
		classifier.fit(X_train, Y_train)

	y_pred_chains = np.array([classifier.predict(X_test) for classifier in chains])

	chain_jaccard_scores = [jaccard_score(Y_test.reshape(-1, 1), y_pred_chain.reshape(-1, 1),
										  average='micro') for y_pred_chain in y_pred_chains]

	fig, ax = plt.subplots(figsize=(7, 4))

	ax.bar(np.arange(len(MODEL_NAMES) - 1), chain_jaccard_scores, alpha=0.5, color='#d63031', width=0.45)

	ax.grid(True)
	ax.set_title('Classifier Chain Performance Comparison')
	ax.set_xticks(np.arange(len(MODEL_NAMES) - 1))
	ax.set_xticklabels(MODEL_NAMES[:-1], rotation='vertical')
	ax.set_ylabel('Jaccard Similarity Score')
	plt.tight_layout()
	plt.show()

#
#
