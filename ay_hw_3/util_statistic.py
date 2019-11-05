# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/25/2019 9:48 PM'

import pandas as pd


def gen_statistic_result(dataFrameObj, fileItemIndex, hasTime=True):
	result = pd.DataFrame()
	startIndex = 1 if hasTime is True else 0
	for index, column in enumerate(dataFrameObj.columns[startIndex:]):
		data = {"min(" + str(index + 1) + ")": dataFrameObj[column].min(),
				"max(" + str(index + 1) + ")": dataFrameObj[column].max(),
				"mean(" + str(index + 1) + ")": dataFrameObj[column].mean(),
				"median(" + str(index + 1) + ")": dataFrameObj[column].median(),
				"standard deviation(" + str(index + 1) + ")": dataFrameObj[column].std(),
				"1st quart(" + str(index + 1) + ")": dataFrameObj[column].quantile(.25),
				"3rd quart(" + str(index + 1) + ")": dataFrameObj[column].quantile(.75)
				}
		temp = pd.DataFrame(data, index=[fileItemIndex])
		result = pd.concat([result, temp], axis=1)

	return result
