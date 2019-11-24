#
from collections import Counter

from tqdm import tqdm

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/30/2019 10:57 PM'

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH
from ay_hw_5.util_data import load_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	X_train, _ = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)

	avg_scores = []
	hamming_dist = []
	k_list = []
	for i in range(2):
		temp_avg_scores = []
		for k in range(2, 10):
			k_means_clf = KMeans(n_clusters=k, random_state=i)
			cluster_labels = k_means_clf.fit_predict(X_train)
			temp_avg_scores.append(silhouette_score(X_train, cluster_labels))
			hamming_dist.append(
				sum(np.min(cdist(X_train, k_means_clf.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])

		temp_best_k = temp_avg_scores.index(max(temp_avg_scores)) + 2
		k_list.append(temp_best_k)
		avg_scores.append(max(temp_avg_scores))

	print("-----------\"The best K\"-------------")
	print("The Best K is : ", Counter(k_list).most_common(1)[0][0])
	print("-----------\"Max Silhouette Score\"-------------")
	print("Max Silhouette Score is : ", max(avg_scores))
	print("-----------\"AVG Hamming Distance\"-------------")
	print("The average of Hamming Distance is : ", np.average(hamming_dist))
	print("-----------\"STD Hamming Distance\"-------------")
	print("The standard deviation of Hamming Distance is : ", np.std(hamming_dist))

	print(avg_scores)
	print(hamming_dist)
	print(k_list)
