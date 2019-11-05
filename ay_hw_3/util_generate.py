# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/26/2019 10:42 PM'

import os
from ay_hw_3._global import FULL_COLUMNS, ROOT_PATH, SPLASH, TIME_DOMAIN
from ay_hw_3.util_data import get_all_datasets_path


def gen_test_data_file_paths(rootPath=ROOT_PATH):
	filePaths = []
	dirs = os.listdir(rootPath)
	for index, dir in enumerate(dirs):
		filePaths.append(rootPath + SPLASH + dir + SPLASH + 'dataset1.csv')
		filePaths.append(rootPath + SPLASH + dir + SPLASH + 'dataset2.csv')
		if index > 1:
			filePaths.append(rootPath + SPLASH + dir + SPLASH + 'dataset3.csv')

	return filePaths


def gen_train_data_file_paths(allFilePaths=None, testFilePaths=None):
	if allFilePaths is None:
		allFilePaths = get_all_datasets_path(ROOT_PATH)
	if testFilePaths is None:
		testFilePaths = gen_test_data_file_paths(ROOT_PATH)

	trainFilePaths = list(set(allFilePaths) - set(testFilePaths))
	trainFilePaths.sort(key=allFilePaths.index)
	return trainFilePaths


def gen_multiple_column_name(parts=2, hasTime=True):
	columns = FULL_COLUMNS if hasTime else TIME_DOMAIN
	names = []
	for index in range(parts):
		names.extend([str(column + "_" + str(index + 1)) for column in columns])

	return names


def gen_multiple_label(parts=6):
	labels = []
	for index in range(parts*6):
		labels.extend(["min(" + str(index + 1) + ")",
					   "max(" + str(index + 1) + ")",
					   "mean(" + str(index + 1) + ")",
					   "median(" + str(index + 1) + ")",
					   "standard deviation(" + str(index + 1) + ")",
					   "1st quart(" + str(index + 1) + ")",
					   "3rd quart(" + str(index + 1) + ")"])

	return labels
