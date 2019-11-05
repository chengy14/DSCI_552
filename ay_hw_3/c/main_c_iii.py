# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/24/2019 8:43 PM'

from ay_hw_3.util_data import load_data_and_label, get_all_datasets_path
from ay_hw_3.util_statistic import gen_statistic_result
from ay_hw_3._global import ROOT_PATH

import pandas as pd
import numpy as np
import pprint

if __name__ == "__main__":
	allFilePaths = get_all_datasets_path(rootPath=ROOT_PATH)

	statisticResult = pd.DataFrame()
	for index, path in enumerate(allFilePaths):
		fileItem, fileLabel = load_data_and_label(path)
		staticResultItem = gen_statistic_result(fileItem, index + 1)
		statisticResult = statisticResult.append(staticResultItem)

	##----Same to the main_c_ii.py-------------------
	confidence_interval = {}
	for column in statisticResult.columns:
		itemCIRange = []
		for i in range(0, 999):
			# Return a random sample of items from an axis of object.
			ran_sample = statisticResult[column].sample(n=10, replace=True)
			stat = ran_sample.std()
			itemCIRange.append(stat)
		itemCIRange.sort()
		lowerValue = np.percentile(itemCIRange, 0.05)
		upperValue = np.percentile(itemCIRange, 0.95)
		confidence_interval[column] = [lowerValue, upperValue]

	pp = pprint.PrettyPrinter(depth=2)
	pp.pprint(confidence_interval)
