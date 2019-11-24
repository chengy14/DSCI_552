#
from collections import Counter
from warnings import simplefilter

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
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

	accuracyList = list()
	precisionList = list()
	recallRateList = list()
	f_scoreList = list()
	aucList = list()
	y_test_predict = None
	y_test_true = None
	for i in tqdm(range(2)):
		X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																			   random_state=i, pos_class=0,
																			   neg_class=1)
		spectral_clf = SpectralClustering(n_clusters=2, affinity="rbf", n_init=10, gamma=1)
		cluster_labels = spectral_clf.fit_predict(X_test)

		indexOfPosDist = np.argwhere(cluster_labels == 0).reshape(-1, )
		indexOfNegDist = np.argwhere(cluster_labels == 1).reshape(-1, )

		y_test_pos = y_test[indexOfPosDist]
		y_test_neg = y_test[indexOfNegDist]
		pos_label = Counter(y_test_pos).most_common(1)[0][0]
		neg_label = Counter(y_test_neg).most_common(1)[0][0]
		cluster_labels[indexOfPosDist] = pos_label
		cluster_labels[indexOfNegDist] = neg_label

		y_test_predict = cluster_labels
		y_test_true = y_test
		precision, recall, f_score, _ = score(y_test_true, y_test_predict, average='binary', pos_label=0)
		precisionList.append(precision)
		recallRateList.append(recall)
		f_scoreList.append(f_score)
		falsePositiveRate, truePositiveRate, _ = roc_curve(y_test_true, y_test_predict)
		aucList.append(auc(falsePositiveRate, truePositiveRate))
		accuracyList.append(accuracy_score(y_test_true, y_test_predict))

	print("-----------\"Overall AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(accuracyList))
	print("AVG Precision: ", np.average(precisionList))
	print("AVG Recall Rate: ", np.average(recallRateList))
	print("AVG F1 -Score: ", np.average(f_scoreList))
	print("AVG AUC: ", np.average(aucList))
	spectralTestDF = pd.DataFrame(data={'Algorithm': ['Spectral Clustering Test']})
	spectralTestDF['AVG Accuracy'] = np.average(accuracyList)
	spectralTestDF['AVG Precision'] = np.average(precisionList)
	spectralTestDF['AVG Recall'] = np.average(recallRateList)
	spectralTestDF['AVG F Score'] = np.average(f_scoreList)
	spectralTestDF['AVG AUC'] = np.average(aucList)
	# infoDF = infoDF.append(spectralTestDF)

	plot_roc(y_test_true, y_test_predict, title="Spectral Clustering ROC")
	print("-----------\"Confusion Matrix about One Run\"-------------")
	print(pd.crosstab(y_test_true, y_test_predict, rownames=['True'], colnames=['Predicted'], margins=True))
