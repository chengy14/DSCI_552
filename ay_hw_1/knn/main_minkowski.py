# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/30/2019 09:31 PM'

# from ay_hw_1.knn.knn_minkowski import KNeighborsClassifier
from ay_hw_1.util import load_data, train_test_by_class_index
from ay_hw_1._global import CLASS0, CLASS1
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

"""
Replace the Euclidean metric with the following metrics and test them. Summarize the test errors (i.e., when k = k∗) in a table. 
Use all of your training data and select the best k when k = {1; 6; 11; : : : ; 196}.
i. Minkowski Distance:
	A. which becomes Manhattan Distance with p = 1.
	
	C. which becomes Chebyshev Distance with p -> infinite
"""

if __name__ == "__main__":

	X_row_data, Y_row_data = load_data('../assets/data.csv')

	split_info_dict = {CLASS0: 70, CLASS1: 140}

	X_train, X_test, y_train, y_test = train_test_by_class_index(X_row_data, Y_row_data, split_info_dict)

	alternative_k = np.arange(1, 197, 5)
	train_accuracy = np.empty(len(alternative_k))
	test_accuracy = np.empty(len(alternative_k))

	for index, k in enumerate(alternative_k):
		# KNeighborsClassifier based on Minkowski Distance
		# knn_clf = KNeighborsClassifier(n_neighbors=k, p=1)	# Manhattan Distance
		# knn_clf = KNeighborsClassifier(n_neighbors=k, p=2)  # Euclidean Distance
		knn_clf = KNeighborsClassifier(n_neighbors=k, p=float('inf'))  # Chebyshev Distance

		knn_clf.fit(X_train, y_train)

		y_predict = knn_clf.predict(X_test)

		train_accuracy[index] = knn_clf.score(X_train, y_train)

		test_accuracy[index] = knn_clf.score(X_test, y_test)

	print(test_accuracy.shape)

	print(alternative_k.shape)
	best_accuracy_ = np.max(test_accuracy)
	best_k_ = (np.reshape(np.argwhere(test_accuracy == best_accuracy_), (1, -1))[0]) * 5 + 1

	plt.title('k-NN Varying number of neighbors (Minkowski P = INF)')
	plt.plot(alternative_k, 1 - test_accuracy, label='Testing Error Rate')
	plt.plot(alternative_k, 1 - train_accuracy, label='Training Error Rate')
	plt.vlines(best_k_, 0, 0.4, colors='red', label='k = ' + str(best_k_), linestyles='dotted')
	plt.hlines(1 - best_accuracy_, 0, 197, colors='gray', label='Error Rate = ' + str(best_accuracy_)[:6],
			   linestyles='dotted')
	plt.legend()
	plt.grid(True)
	plt.xlabel('Number of neighbors')
	plt.ylabel('Error Rate')
	plt.show()
