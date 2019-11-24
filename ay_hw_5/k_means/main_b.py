#
from sklearn.metrics import silhouette_score

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/30/2019 10:57 PM'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH, LABELS_NAME
from ay_hw_5.util_data import load_data

PREDICT_LABEL = 'predicted'

if __name__ == "__main__":
	X_train, y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	y_data = pd.DataFrame(y_data, columns=LABELS_NAME)

	avg_scores = []
	for i in range(1, 51):
		temp_avg_scores = []
		predicted_results = []
		for k in range(2, 10):
			k_means_clf = KMeans(n_clusters=k, random_state=i)
			predicted_labels = k_means_clf.fit_predict(X_train)
			predicted_results.append(predicted_labels)
			temp_avg_scores.append(silhouette_score(X_train, predicted_labels))

		temp_best_k = temp_avg_scores.index(max(temp_avg_scores)) + 2
		majority_label = dict()
		y_data[PREDICT_LABEL] = predicted_results[np.argmax(temp_avg_scores)]

		for class_index in range(temp_best_k):
			matched_label_data = y_data[y_data[PREDICT_LABEL] == class_index]
			temp = {}
			for column in LABELS_NAME:
				temp[column] = Counter(matched_label_data[column]).most_common(1)[0][0]
			majority_label[class_index] = temp

		print("in {} th loop, the best k is {}, and the majority class in each label is : {}".format(i, temp_best_k,
																									 majority_label))
