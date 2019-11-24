#
from ay_hw_6._global import ROOT_PATH, WDBC_FILE_PATH, SPLASH
from ay_hw_6.util_data import load_data, train_test_split_by_class_and_ratio

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/5/2019 11:57 AM'

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + SPLASH + WDBC_FILE_PATH)
	X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																		   random_state=2333)

	print("-----------\"Row Data\"-------------")
	print("the shape of X_data is: ", X_data.shape)
	print("the shape of y_data is: ", y_data.shape)
	print("-----------\"After Split\"-------------")
	print("the shape of X_train is: ", X_train.shape)
	print("the shape of X_test is: ", X_test.shape)
	print("the shape of y_train is: ", y_train.shape)
	print("the shape of y_test is: ", y_test.shape)
