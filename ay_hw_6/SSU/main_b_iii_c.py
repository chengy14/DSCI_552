#
from collections import Counter
from warnings import simplefilter

import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm

from ay_hw_6.util_plot import plot_roc

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/6/2019 8:52 AM'

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as score
from ay_hw_6._global import ROOT_PATH, WDBC_FILE_PATH, SPLASH, M
from ay_hw_6.util_data import load_data, train_test_split_by_class_and_ratio


def checkResult(labels):
	majorityPredClass = Counter(labels).most_common(1)[0][0]
	if majorityPredClass != 0:
		flip = np.vectorize(lambda x: 1 if x == 0 else 0)
		return flip(labels).tolist()
	return labels


if __name__ == "__main__":
	simplefilter(action='ignore', category=ConvergenceWarning)
	X_data, y_data = load_data(ROOT_PATH + SPLASH + WDBC_FILE_PATH)
	X_data = MinMaxScaler().fit(X_data).transform(X_data)
	labelEncoder = LabelEncoder().fit(['B', 'M'])
	y_data = labelEncoder.transform(y_data)

	precisionList = list()
	recallRateList = list()
	f_scoreList = list()
	aucList = list()
	y_test_predict = list()
	accuracyList = list()
	y_test_true = None
	for i in tqdm(range(M)):
		X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																			   random_state=i, pos_class=0,
																			   neg_class=1)

		k_means_clf = KMeans(n_clusters=2, init='random', n_init=10)
		cluster_labels = k_means_clf.fit_predict(X_test)
		posDist = cdist(X_test, k_means_clf.cluster_centers_, 'euclidean')[:, 0]
		negDist = cdist(X_test, k_means_clf.cluster_centers_, 'euclidean')[:, 1]
		y_test_predict.clear()
		for index, pred_label in enumerate(cluster_labels):
			if pred_label == 0:
				y_test_predict.append(0 if abs(posDist[index]) < abs(negDist[index]) else 1)
			else:
				y_test_predict.append(1 if abs(posDist[index]) > abs(negDist[index]) else 0)

		y_test_predict = checkResult(y_test_predict)
		y_test_true = y_test
		precision, recall, f_score, _ = score(y_test, y_test_predict, average='binary', pos_label=0)
		precisionList.append(precision)
		recallRateList.append(recall)
		f_scoreList.append(f_score)
		falsePositiveRate, truePositiveRate, _ = roc_curve(y_test, y_test_predict)
		aucList.append(auc(falsePositiveRate, truePositiveRate))
		accuracyList.append(accuracy_score(y_test, y_test_predict))

	print("-----------\"Overall AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(accuracyList))
	print("AVG Precision: ", np.average(precisionList))
	print("AVG Recall Rate: ", np.average(recallRateList))
	print("AVG F1 -Score: ", np.average(f_scoreList))
	print("AVG AUC: ", np.average(aucList))
	unSupTestDF = pd.DataFrame(data={'Algorithm': ['UnSupervised Test']})
	unSupTestDF['AVG Accuracy'] = np.average(accuracyList)
	unSupTestDF['AVG Precision'] = np.average(precisionList)
	unSupTestDF['AVG Recall'] = np.average(recallRateList)
	unSupTestDF['AVG F Score'] = np.average(f_scoreList)
	unSupTestDF['AVG AUC'] = np.average(aucList)
	# infoDF = infoDF.append(unSupTestDF)
	plot_roc(y_test_true, y_test_predict, title="K-means ROC")
	print("-----------\"Confusion Matrix about One Run\"-------------")
	print(pd.crosstab(y_test_true, np.reshape(y_test_predict, (-1,)), rownames=['True'], colnames=['Predicted'], margins=True))
