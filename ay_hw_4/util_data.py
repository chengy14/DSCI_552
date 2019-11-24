#

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 10:14 AM'

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ay_hw_4._global import CRIME_FULL_COLUMNS


def genderate_column_indexes(dataFrame, y_column_index):
	x_column_index = list(range(0, dataFrame.shape[1]))
	if y_column_index == -1:
		x_column_index.remove(dataFrame.shape[1] - 1)
	else:
		x_column_index.remove(y_column_index)

	return x_column_index


def data_imputation(dataFrame, assignedColumnNames, y_column_name):
	# get the indexes if the column's content has nan
	imputeIndex = list()
	missingValueColumn = dataFrame.columns[dataFrame.isnull().any()]
	remainColumnNames = list(set(assignedColumnNames) - set(missingValueColumn.tolist()))
	remainColumnNames.remove(y_column_name)
	remainColumnNames.sort(key=assignedColumnNames.index)
	for index in missingValueColumn:
		# if there are more than 2/3 of data exist in that column, we can use mean strategy to impute data.
		# print(X_data[index].item())
		if dataFrame[index].count() > (2 * len(dataFrame) / 3):
			imputeIndex.append(index)
			remainColumnNames.append(index)

	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	for index in imputeIndex:
		originalValue = dataFrame[index].to_numpy().reshape(-1, 1)
		imp_mean.fit(originalValue)
		dataFrame[index] = imp_mean.transform(originalValue)

	return dataFrame, remainColumnNames


def load_data(filePath, skip_first_column=0, skip_first_row=0, y_column_index=-1, needImpute=False,
			  assignedColumnNames=CRIME_FULL_COLUMNS, missingSymbol='?', dropOrNot=True, get_first_rows=None):
	assert (filePath is not None) and (len(filePath) >= 1), "empty file path"
	dataFrame = pd.read_csv(filePath, encoding='utf-8', delimiter=",", header=None, skiprows=skip_first_row)
	dataFrame.columns = assignedColumnNames
	y_column_name = assignedColumnNames[y_column_index]

	# skip some columns
	dataFrame.drop(dataFrame.iloc[:, :skip_first_column], inplace=True, axis=1)
	assignedColumnNames = assignedColumnNames[skip_first_column:]

	x_column_index = genderate_column_indexes(dataFrame, y_column_index)

	# If need process data imputation
	if needImpute:

		X_data = dataFrame.iloc[:, x_column_index].replace(missingSymbol, np.nan)
		# get the indexes if the column's content has nan
		X_data, remainColumnNames = data_imputation(X_data, assignedColumnNames, y_column_name)

		# drop some column which contain missing value.
		if dropOrNot:
			X_data = X_data.dropna(axis=1)
			X_data.columns = remainColumnNames
		y_data = dataFrame.iloc[:, y_column_index]
	else:
		X_data = dataFrame.iloc[:, x_column_index]
		y_data = dataFrame.iloc[:, y_column_index]

	return X_data, y_data


def train_test_split_by_size(X_data, y_data, train_size, random_state=None):
	assert X_data.shape[0] == y_data.shape[0], \
		"the size of X must be equal to the size of y"
	if random_state:
		np.random.seed(random_state)

	shuffled_indexes = np.random.permutation(len(X_data))

	test_size = len(X_data) - train_size
	test_indexes = shuffled_indexes[:test_size]
	train_indexes = shuffled_indexes[test_size:]

	X_train = X_data.loc[train_indexes, :]
	y_train = y_data.loc[train_indexes]

	X_test = X_data.loc[test_indexes, :]
	y_test = y_data.loc[test_indexes]

	return X_train, X_test, y_train, y_test


def to_binary_numeric(y_data, classNeg="neg"):
	if isinstance(y_data, pd.DataFrame):
		y_data = y_data.to_numpy()

	y_data = np.array([1 if x == classNeg else 0 for x in y_data])
	return pd.DataFrame(y_data)
