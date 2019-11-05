# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/24/2019 1:30 PM'

from ay_hw_3.util_data import load_data_and_label, get_all_datasets_path,is_bending
from ay_hw_3.util_statistic import gen_statistic_result
from ay_hw_3._global import ROOT_PATH

import pandas as pd

if __name__ == "__main__":
	allFilePaths = get_all_datasets_path(rootPath=ROOT_PATH)
	statisticResult = pd.DataFrame()
	for index, path in enumerate(allFilePaths):
		# get each file content and file label
		fileItem, fileLabel = load_data_and_label(path)
		# based on the data saved in file, we calculate the 6 statistic term(time-domain feature)
		staticResultItem = gen_statistic_result(fileItem, index + 1)
		staticResultItem["label"] = is_bending(fileLabel)
		statisticResult = statisticResult.append(staticResultItem)

	print("-----------------\"Head of the Time-Domain DataSet\"----------------")
	print(statisticResult.head())
	print("---\"Shape of the Time-Domain DataSet\"----")
	print("Time-Domain DataSet Shape: ", statisticResult.shape)