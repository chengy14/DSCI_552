#
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, hamming_loss

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/30/2019 10:57 PM'

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, LABELS_NAME
from ay_hw_5.util_data import load_data

PREDICT_LABEL = 'predicted'

if __name__ == "__main__":
	X_train, y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	y_data = pd.DataFrame(y_data, columns=LABELS_NAME)
	hamming_loss_list = list()
	hamming_dist = list()
	avg_scores = []
	for i in range(1, 10):
		temp_avg_scores = []
		predicted_results = []
		for k in range(3, 5):
			k_means_clf = KMeans(n_clusters=k, random_state=i)
			predicted_labels = k_means_clf.fit_predict(X_train)
			predicted_results.append(predicted_labels)
			temp_avg_scores.append(silhouette_score(X_train, predicted_labels))
			hamming_dist.append(
				sum(np.min(cdist(X_train, k_means_clf.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])

		temp_best_k = temp_avg_scores.index(max(temp_avg_scores)) + 2
		majority_label = dict()
		y_data[PREDICT_LABEL] = predicted_results[np.argmax(temp_avg_scores)]

		for class_index in range(temp_best_k):
			matched_label_data = y_data[y_data[PREDICT_LABEL] == class_index]
			temp = {}
			for column in LABELS_NAME:
				temp[column] = Counter(matched_label_data[column]).most_common(1)[0][0]
			majority_label[class_index] = temp

		unmatched = 0
		for class_index in range(temp_best_k):
			matched_label_data = y_data[y_data[PREDICT_LABEL] == class_index]
			for column in LABELS_NAME:
				unmatched += sum(matched_label_data[column] != majority_label[class_index][column])
		hamming_loss = unmatched / (y_data.shape[0] * 3)
		hamming_loss_list.append(hamming_loss)

	print("-----------\"AVG Hamming Distance\"-------------")
	print("The average of Hamming Distance is : ", np.average(hamming_dist))
	print("-----------\"AVG Hamming Score\"-------------")
	print("The average of Hamming Score is : ", np.average(1 - np.array(hamming_loss_list)))
	print("-----------\"AVG Hamming Loss\"-------------")
	print("The average of Hamming Loss is : ", np.average(hamming_loss_list))
