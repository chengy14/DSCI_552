# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/4/2019 10:10 PM'

import pandas as pd

from ay_hw_4._global import ROOT_PATH, APS_TRAIN, APS_TEST, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data
from ay_hw_4.util_stratistic import count_neg_and_pos

if __name__ == "__main__":
	pd.set_option('display.max_columns', 100)
	X_train, y_train = load_data(ROOT_PATH + APS_TRAIN, skip_first_row=21, y_column_index=0,
								 assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
								 dropOrNot=False)

	X_test, y_test = load_data(ROOT_PATH + APS_TEST, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True,
							   dropOrNot=False)

	train_num_pos, train_num_neg = count_neg_and_pos(y_train)
	test_num_pos, test_num_neg = count_neg_and_pos(y_test)
	print("the number of pos data is : ", train_num_pos + test_num_pos)
	print("the number of neg data is : ", train_num_neg + test_num_neg)
