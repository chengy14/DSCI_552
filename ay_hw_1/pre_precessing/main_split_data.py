# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 10:17 AM'

from ay_hw_1.util import load_data, train_test_by_class_index
from ay_hw_1._global import CLASS0, CLASS1
import pandas as pd

'''
iii. Select the first 70 rows of Class 0 and the first 140 rows of Class 1 as the
training set and the rest of the data as the test set
'''

if __name__ == "__main__":
	X_row_data, Y_row_data = load_data('../assets/data.csv')

	split_info_dict = {CLASS0: 70, CLASS1: 140}

	X_train, X_test, y_train, y_test = \
		train_test_by_class_index(X_row_data, Y_row_data, split_info_dict)

	print("X_train.shape = {}".format(X_train.shape))
	print("X_test.shape = {}".format(X_test.shape))
	print("y_train.shape = {}".format(y_train.shape))
	print("y_test.shape = {}".format(y_test.shape))

