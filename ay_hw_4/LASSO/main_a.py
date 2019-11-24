# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/2/2019 9:10 AM'

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data, train_test_split_by_size

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, y_column_index=-1)

	print("X Row Data Shape: ", X_data.shape)
	print("y Row Data Shape: ", y_data.shape)
	X_train, X_test, y_train, y_test = train_test_split_by_size(X_data, y_data, train_size=1495, random_state=2333)
	print("X_Train Data Shape: ", X_train.shape)
	print("y_Train Data Shape: ", y_train.shape)
	print("X_test Data Shape: ", X_test.shape)
	print("y_test Data Shape: ", y_test.shape)
