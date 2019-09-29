# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 11:31 AM'

# from ay_hw_1.knn.knn_euclidean import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from ay_hw_1.util import load_data, train_test_by_class_index
from ay_hw_1._global import CLASS0, CLASS1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd

"""
Calculate the confusion matrix, true positive rate, true negative rate, precision,
and F-score when k = k∗
"""

if __name__ == "__main__":
	X_row_data, Y_row_data = load_data('../assets/data.csv')

	split_info_dict = {CLASS0: 70, CLASS1: 140}

	X_train, X_test, y_train, y_test = train_test_by_class_index(X_row_data, Y_row_data, split_info_dict)

	knn_clf = KNeighborsClassifier(n_neighbors=4)
	knn_clf.fit(X_train, y_train)
	y_predict = knn_clf.predict(X_test)

	crosstab = pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'], margins=True)
	true_positive = crosstab.loc[[0]].values[0][0]
	false_positive = crosstab.loc[[0]].values[0][1]
	false_negative = crosstab.loc[[1]].values[0][0]
	true_negative = crosstab.loc[[1]].values[0][1]
	print("Sklearn-Euclidean Distance k = ", 4)
	print("-----------------------------------------------------")
	print(crosstab)
	print("-----------------------------------------------------")

	print("true_positive_rate(TPR) = ", true_positive / (true_positive + false_negative))
	print("true_negative_rate(TNR) = ", true_negative / (false_positive + true_negative))
	print("-----------------------------------------------------")

	print(classification_report(y_test, y_predict))
