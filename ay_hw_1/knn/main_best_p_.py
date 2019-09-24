# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/30/2019 09:31 PM'

# from ay_hw_1.knn.knn_minkowski import KNeighborsClassifier
from ay_hw_1.util import load_data, train_test_by_class_index
from ay_hw_1._global import CLASS0, CLASS1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import math

"""
Replace the Euclidean metric with the following metrics and test them. Summarize the test errors (i.e., when k = k∗) in a table. 
Use all of your training data and select the best k when k = {1; 6; 11; : : : ; 196}.
i. Minkowski Distance:
	
	B. with log10(p)  = {0.1; 0.2; 0.3; . . . 1}. In this case, use the k∗ you found
		for the Manhattan distance in 1(d)iA. What is the best log10(p)?
	
"""

if __name__ == "__main__":

	X_row_data, Y_row_data = load_data('../assets/data.csv')

	split_info_dict = {CLASS0: 70, CLASS1: 140}

	X_train, X_test, y_train, y_test = train_test_by_class_index(X_row_data, Y_row_data, split_info_dict)

	alternative_p = [10 ** p for p in np.arange(0.1, 1.1, 0.1)]
	train_accuracy = np.empty(len(alternative_p))
	test_accuracy = np.empty(len(alternative_p))

	for index, p in enumerate(alternative_p):
		# KNeighborsClassifier based on Minkowski Distance
		# in both sk_minkowski_p1_accurancy and minkowski_p1_accurancy pics, the best k = [1 6 11 26]
		knn_clf = KNeighborsClassifier(n_neighbors=6, p=p)

		knn_clf.fit(X_train, y_train)

		y_predict = knn_clf.predict(X_test)

		train_accuracy[index] = knn_clf.score(X_train, y_train)

		test_accuracy[index] = knn_clf.score(X_test, y_test)

	best_accuracy_ = np.max(test_accuracy)
	best_p_ = 10 ** (np.argmax(test_accuracy) * 0.1 + 0.1)

	plt.title('k-NN Varying Value of Argument P (Sklearn-Minkowski k = 6 P = {})'.format(str(best_p_)[:6]))
	plt.plot(alternative_p, 1 - test_accuracy, label='Testing Error Rate')
	plt.plot(alternative_p, 1 - train_accuracy, label='Training Error Rate')
	plt.vlines(best_p_, 0, 0.4, colors='red', label='p = ' + str(best_p_), linestyles='dotted')
	plt.hlines(1 - best_accuracy_, 1, 10, colors='gray', label='error rate = ' + str(1 - best_accuracy_)[:6],
			   linestyles='dotted')
	plt.legend()
	plt.grid(True)
	plt.xlabel('Value of Argument P')
	plt.ylabel('Error Rate')
	plt.show()

