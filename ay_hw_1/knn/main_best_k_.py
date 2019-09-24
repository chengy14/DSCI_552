# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 11:31 AM'

from ay_hw_1.knn.knn_euclidean import KNeighborsClassifier
from ay_hw_1.util import load_data, train_test_by_class_index
from ay_hw_1._global import CLASS0, CLASS1
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

"""Test all the data in the test database with k nearest neighbors. Take decisions by majority polling. 
Plot train and test errors in terms of k for k from {208; 205; : : : ; 7; 4; 1;} (in reverse order). 
You are welcome to use smaller increments of k. 
Which k∗ is the most suitable k among those values? 
"""

if __name__ == "__main__":

	X_row_data, Y_row_data = load_data('../assets/data.csv')

	split_info_dict = {CLASS0: 70, CLASS1: 140}

	X_train, X_test, y_train, y_test = train_test_by_class_index(X_row_data, Y_row_data, split_info_dict)

	# X_train, X_test, y_train, y_test = train_test_split(X_row_data, Y_row_data, test_size=0.3, random_state=666)

	alternative_k = np.arange(208, 0, -3)
	train_accuracy = np.empty(len(alternative_k))
	test_accuracy = np.empty(len(alternative_k))

	for index, k in enumerate(alternative_k):
		# print("{} -- {}".format(index, k))
		# KNeighborsClassifier based on Euclidean Distance
		knn_clf = KNeighborsClassifier(n_neighbors=k)

		knn_clf.fit(X_train, y_train)

		train_accuracy[index] = knn_clf.score(X_train, y_train)

		test_accuracy[index] = knn_clf.score(X_test, y_test)

	best_accuracy_ = np.max(test_accuracy)
	best_k_ = 208 - (np.reshape(np.argwhere(test_accuracy == best_accuracy_), (1, -1))[0]) * 3
	plt.title('k-NN Varying number of neighbors (Aaron-Euclidean)')
	plt.plot(alternative_k, 1 - test_accuracy, label='Testing Error Rate')
	plt.plot(alternative_k, 1 - train_accuracy, label='Training Error Rate')
	plt.vlines(best_k_, 0, 0.4, colors='red', label='k = ' + str(best_k_),linestyles='dotted')
	plt.hlines(1 - best_accuracy_, 0, 208, colors='gray', label='error rate = ' + str(1 - best_accuracy_)[:5], linestyles='dotted')
	plt.legend()
	plt.grid(True)
	plt.xlabel('Number of neighbors')
	plt.ylabel('Error Rate')
	plt.show()


