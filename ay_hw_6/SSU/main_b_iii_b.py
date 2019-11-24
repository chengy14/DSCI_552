#
from collections import Counter
from warnings import simplefilter

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import pandas as pd
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
	accuracyList = list()
	y_predict = None
	y_true = None
	for i in tqdm(range(M)):
		X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																			   random_state=i, pos_class=0,
																			   neg_class=1)
		k_means_clf = KMeans(n_clusters=2, init='random', n_init=10)
		cluster_labels = k_means_clf.fit_predict(X_train)
		indexOfPosDist = np.argsort(cdist(X_train, k_means_clf.cluster_centers_, 'euclidean')[:, 0], axis=0)[:30]
		indexOfNegDist = np.argsort(cdist(X_train, k_means_clf.cluster_centers_, 'euclidean')[:, 1], axis=0)[:30]

		y_train_pos_nearest_30 = y_train[indexOfPosDist]
		y_train_neg_nearest_30 = y_train[indexOfNegDist]
		pos_label = Counter(y_train_pos_nearest_30).most_common(1)[0][0]
		neg_label = Counter(y_train_neg_nearest_30).most_common(1)[0][0]
		majorityPolling = np.vectorize(lambda x: pos_label if x == 0 else neg_label)
		new_cluster_labels = majorityPolling(cluster_labels)
		y_predict = new_cluster_labels
		y_true = y_train
		precision, recall, f_score, _ = score(y_train, y_predict, average='binary', pos_label=0)
		accuracyList.append(accuracy_score(y_train, y_predict))
		precisionList.append(precision)
		recallRateList.append(recall)
		f_scoreList.append(f_score)
		falsePositiveRate, truePositiveRate, _ = roc_curve(y_train, y_predict)
		aucList.append(auc(falsePositiveRate, truePositiveRate))

	print("-----------\"Overall AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(accuracyList))
	print("AVG Precision: ", np.average(precisionList))
	print("AVG Recall Rate: ", np.average(recallRateList))
	print("AVG F1 -Score: ", np.average(f_scoreList))
	print("AVG AUC: ", np.average(aucList))
	unSupTrainDF = pd.DataFrame(data={'Algorithm': ['UnSupervised Train']})
	unSupTrainDF['AVG Accuracy'] = np.average(accuracyList)
	unSupTrainDF['AVG Precision'] = np.average(precisionList)
	unSupTrainDF['AVG Recall'] = np.average(recallRateList)
	unSupTrainDF['AVG F Score'] = np.average(f_scoreList)
	unSupTrainDF['AVG AUC'] = np.average(aucList)
	# infoDF = infoDF.append(unSupTrainDF)
	plot_roc(y_true, y_predict, title="K-means ROC")
	print("-----------\"Confusion Matrix about One Run\"-------------")
	print(pd.crosstab(y_true, y_predict, rownames=['True'], colnames=['Predicted'], margins=True))
