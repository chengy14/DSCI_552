#
from warnings import simplefilter

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc, accuracy_score
from tqdm import tqdm

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/6/2019 8:52 AM'

import numpy as np
import pandas as pd
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
	accuracyList = list()
	aucList = list()
	for i in tqdm(range(M)):
		X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																			   random_state=i, pos_class=0, neg_class=1)
		k_means_clf = KMeans(n_clusters=2, init='random', n_init=10)
		cluster_labels = k_means_clf.fit_predict(X_train)
		precision, recall, f_score, _ = score(y_train, cluster_labels, average='binary', pos_label=0)
		precisionList.append(precision)
		recallRateList.append(recall)
		f_scoreList.append(f_score)
		accuracyList.append(accuracy_score(y_train, cluster_labels))
		falsePositiveRate, truePositiveRate, _ = roc_curve(y_train, cluster_labels)
		aucList.append(auc(falsePositiveRate, truePositiveRate))

	print("-----------\"STD Infos\"-------------")
	print("MAX Accuracy Score: ", max(accuracyList))
	print("MAX Precision: ", max(precisionList))
	print("MAX Recall Rate: ", max(recallRateList))
	print("MAX F1 -Score: ", max(f_scoreList))
	print("MAX AUC: ", max(aucList))

