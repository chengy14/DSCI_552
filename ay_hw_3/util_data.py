#

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 10:14 AM'

import pandas as pd
import os
from ay_hw_3._global import FULL_COLUMNS, TIME_DOMAIN, DATASET_LABEL
from past.translation import splitall


def split_DF_in_parts(datFrameObj, parts=2, needConcat=True):
	result = pd.DataFrame()
	division = len(datFrameObj) / float(parts)
	# list = np.array_split(datFrameObj.to_numpy(), parts)
	list = [datFrameObj.to_numpy()[int(round(division * i)): int(round(division * (i + 1)))] for i in range(parts)]
	if needConcat:

		for index in range(len(list)):
			result = pd.concat([result, pd.DataFrame(list[index])], axis=1, sort=False)
	else:
		result = []
		for index in range(len(list)):
			result.append(pd.DataFrame(list[index]))
	return result


def load_data_and_label(filePath, columns=FULL_COLUMNS, hasTime=True):
	assert (filePath is not None) and (len(filePath) >= 1), "empty file path"
	if hasTime is False:
		columns = TIME_DOMAIN
	# >>> splitall('C:\\a\\b')   ==>  ['C:\\', 'a', 'b']
	label = splitall(filePath)[2]
	dataframe = pd.read_csv(filePath, encoding='utf-8', delimiter=",", skiprows=5, names=columns)
	return dataframe, label


def load_multiple_files(filePaths, columns=FULL_COLUMNS, hasTime=True):
	"""read records from csv files"""
	assert (filePaths is not None) and (len(filePaths) >= 0), "empty file path"
	if hasTime is False:
		columns = TIME_DOMAIN
	dataFrameObj = pd.DataFrame()
	for index, path in enumerate(filePaths):
		fileItem, fileLabel = load_data_and_label(path, columns=columns)
		dataFrameObj = dataFrameObj.append(fileItem)

	return dataFrameObj


def get_all_datasets_path(rootPath):
	filePaths = []
	for path, directories, files in os.walk(rootPath):
		for file in files:
			filePaths.append(os.path.join(path, file))
	return filePaths


def convert_label_2_num(label):
	return DATASET_LABEL[str.upper(label)].value

def is_bending(label):
	return 1 if label == "bending1" or label == 'bending2' else 0
