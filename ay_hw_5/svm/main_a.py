# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/26/2019 3:19 PM'

from ay_hw_5.util_data import load_data, train_test_split_by_ratio
from ay_hw_5._global import ROOT_PATH, SPLASH, MFCC_FILE_PATH

if __name__ == "__main__":
	X_data, Y_data = load_data(ROOT_PATH + SPLASH + MFCC_FILE_PATH)
	X_train, X_test, Y_train, Y_test = train_test_split_by_ratio(X_data, Y_data, test_size=0.3, random_state=2333)
	print("X_train's shape: ", X_train.shape)
	print("X_test's shape: ", X_test.shape)
	print("Y_train's shape: ", Y_train.shape)
	print("Y_test's shape: ", Y_test.shape)
