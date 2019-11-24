# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/4/2019 3:16 PM'

from ay_hw_4._global import ROOT_PATH, APS_TRAIN, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data

if __name__ == "__main__":

	X_data, y_data = load_data(ROOT_PATH + APS_TRAIN, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True, dropOrNot=False)
	print(X_data.head())
	print(y_data.head())

	# existence larger than 2/3 use mean strategy
	missing_columns = X_data.columns[X_data.isnull().any()]
	print(X_data[missing_columns].describe())
