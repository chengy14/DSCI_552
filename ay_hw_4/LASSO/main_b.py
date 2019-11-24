# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/2/2019 6:39 PM'

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data

if __name__ == "__main__":
	pd.set_option('display.max_columns', 100)
	X_data, y_data = load_data(ROOT_PATH + CRIME, y_column_index=-1, skip_first_column=5)
	print("X_data Row Data Shape: ", X_data.shape)
	print("y Row Data Shape: ", y_data.shape)
	X_data = X_data.replace('?', np.nan)
	missingValueColumnIndex = X_data.columns[X_data.isnull().any()]

	print("In the beginning, there are total {} columns has missing value in the dataset ".format(
		missingValueColumnIndex.shape[0]))
	print("------------------------------------------------------------------------------")
	print(X_data[missingValueColumnIndex].describe())

	# so  we can only impute only one column (index=25)
	index = 'OtherPerCap'
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(X_data[[index]])
	X_data[[index]] = imp_mean.transform(X_data[[index]])
	print("------------------------------------------------------------------------------")
	missingValueColumnIndex = X_data.columns[X_data.isnull().any()]
	# remain = list(set(REMAIN_COLUMNS) - set(missingValueColumnIndex))
	# remain.sort(key=REMAIN_COLUMNS.index)
	# print(remain)
	# print("------------------------------------------------------------------------------")
	print("After Data Imputation (mean strategy), we only have {} columns has missing value. ".format(
		missingValueColumnIndex.shape[0]))

	# drop some column which contain missing value.
	X_data = X_data.dropna(axis=1)
	print("Finally, I gonna use X_data in the shape of {} to training algorithm.".format(X_data.shape))
