# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 5:25 PM'

from ay_hw_3.util_generate import gen_test_data_file_paths, gen_multiple_column_name
from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts,is_bending
from ay_hw_3.util_statistic import gen_statistic_result
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
import pandas as pd

if __name__ == "__main__":
	simplefilter(action='ignore', category=FutureWarning)
	# the best l I got is 3
	bestL = 3
	selectedFeatures = ['min(5)', '3rd quart(5)', '3rd quart(7)', 'max(18)']
	allTestFilePaths = gen_test_data_file_paths()
	# based on what the pdf said, we need to use all training data
	statisticResult = pd.DataFrame()
	for index, path in enumerate(allTestFilePaths):
		fileItem, fileLabel = load_data_and_label(path, hasTime=False)
		gluedFile = split_DF_in_parts(fileItem, parts=bestL, needConcat=True)
		gluedFile.columns = gen_multiple_column_name(parts=bestL, hasTime=False)
		staticResultItem = gen_statistic_result(gluedFile, index + 1, hasTime=False)
		staticResultItem["label"] = is_bending(fileLabel)
		statisticResult = statisticResult.append(staticResultItem, sort=False)

	X_testData = statisticResult[selectedFeatures]
	y_testData = statisticResult['label']
	#
	logitModel = LogisticRegression()
	logitModel.fit(X_testData, y_testData)
	y_predict = logitModel.predict(X_testData)
	print(logitModel.score(X_testData, y_testData))
