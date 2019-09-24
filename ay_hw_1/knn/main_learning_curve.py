# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 11:31 AM'

from ay_hw_1.util import load_data, train_test_by_class_index
from ay_hw_1._global import CLASS0, CLASS1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import math

"""Test all the data in the test database with k nearest neighbors. Take decisions by majority polling. 
Plot train and test errors in terms of k for k from {208; 205; : : : ; 7; 4; 1;} (in reverse order). 
You are welcome to use smaller increments of k. 
Which k∗ is the most suitable k among those values? 
"""

if __name__ == "__main__":

	X_row_data, Y_row_data = load_data('../assets/data.csv')

	alt_N = np.arange(10, 211, 10)
	test_accuracy_temp = np.zeros(21)
	for index, N in enumerate(alt_N):
		class0_row_num_ = math.floor(N / 3)
		class1_row_num_ = N - class0_row_num_
		split_info_dict_ = {CLASS0: class0_row_num_, CLASS1: class1_row_num_}
		X_train, X_test, y_train, y_test = train_test_by_class_index(X_row_data, Y_row_data, split_info_dict_)

		# Standardization
		standardScaler = StandardScaler()
		standardScaler.fit(X_train)
		X_train = standardScaler.transform(X_train)
		X_test = standardScaler.transform(X_test)

		for k in np.arange(1, N, 5):
			knn_clf = KNeighborsClassifier(n_neighbors=k)
			knn_clf.fit(X_train, y_train)
			score = knn_clf.score(X_test, y_test)
			# print("N = {}, k = {}, score = {}".format(N, k, score))

			test_accuracy_temp[index] = score if test_accuracy_temp[index] < score else test_accuracy_temp[index]

	test_error_rate_ = np.array([1 - score for score in test_accuracy_temp])

	higest_accuracy_rate_ = np.max(test_accuracy_temp)
	lowest_accuracy_rate_ = np.min(test_accuracy_temp)
	higest_accuracy_N_ = np.argmax(test_accuracy_temp) * 10 + 10
	lowest_accuracy_N_ = np.argmin(test_accuracy_temp) * 10 + 10

	plt.title('Learning Curve (Sklearn)')
	plt.plot(alt_N, test_error_rate_, label='Error Rate')
	plt.vlines(higest_accuracy_N_, 0, 0.35, colors='red', label='N = ' + str(higest_accuracy_N_), linestyles='dotted')
	plt.hlines(1 - higest_accuracy_rate_, 0, 211, colors='red', label='error rate = ' + str(1 - higest_accuracy_rate_),
			   linestyles='dotted')

	plt.vlines(lowest_accuracy_N_, 0, 0.35, colors='green', label='N = ' + str(lowest_accuracy_N_), linestyles='dotted')
	plt.hlines(1 - lowest_accuracy_rate_, 0, 211, colors='green', label='error rate = ' + str(1 - lowest_accuracy_rate_),
			   linestyles='dotted')
	plt.legend()
	plt.grid(True)
	plt.xlabel('Size of Training Sets')
	plt.ylabel('Error Rate')
	plt.show()

