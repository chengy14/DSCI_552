#
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 7:04 PM'

from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts, convert_label_2_num
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_column_name, gen_test_data_file_paths
from ay_hw_3.util_statistic import gen_statistic_result

if __name__ == "__main__":

	simplefilter(action='ignore', category=FutureWarning)
	allTrainFilePaths = gen_train_data_file_paths()
	allTestFilePaths = gen_test_data_file_paths()
	# based on what the pdf said, we need to use all training data

	trainStaticResult = pd.DataFrame()
	testStaticResult = pd.DataFrame()
	gaussianTestErrorRateList = list()

	for parts in range(1, 10):
		for index, path in enumerate(allTrainFilePaths):
			trainFileItem, trainFileLabel = load_data_and_label(path, hasTime=False)
			gluedTrainFile = split_DF_in_parts(trainFileItem, parts=parts, needConcat=True)
			gluedTrainFile.columns = gen_multiple_column_name(parts=parts, hasTime=False)
			trainStaticResultItem = gen_statistic_result(gluedTrainFile, index + 1, hasTime=False)
			trainStaticResultItem["label"] = convert_label_2_num(trainFileLabel)
			trainStaticResult = trainStaticResult.append(trainStaticResultItem, sort=False)

		for index, path in enumerate(allTestFilePaths):
			testFileItem, testFileLabel = load_data_and_label(path, hasTime=False)
			gluedTestFile = split_DF_in_parts(testFileItem, parts=parts, needConcat=True)
			gluedTestFile.columns = gen_multiple_column_name(parts=parts, hasTime=False)
			testStaticResultItem = gen_statistic_result(gluedTestFile, index + 1, hasTime=False)
			testStaticResultItem["label"] = convert_label_2_num(testFileLabel)
			testStaticResult = testStaticResult.append(testStaticResultItem, sort=False)

		X_train = trainStaticResult[list(trainStaticResult.columns.values)[:-1]]
		y_train = trainStaticResult['label']

		X_test = testStaticResult[list(testStaticResult.columns.values)[:-1]]
		y_test = testStaticResult['label']

		multinomialClassifier = MultinomialNB()
		gaussianTestErrorRateList.append(1 - np.mean(cross_val_score(multinomialClassifier, X_train, y_train, cv=5)))

		# allocate a new space for statisticResult
		trainStaticResult = pd.DataFrame()
		testStaticResult = pd.DataFrame()

	print("-----------\"Test Error Rate\"-------------")
	print("Test Error Rate : ", gaussianTestErrorRateList)
	# # find the best l
	bestL = gaussianTestErrorRateList.index(min(gaussianTestErrorRateList)) + 1
	# bestL = 4
	print("-----------\"Best L\"-------------")
	print("Best l : ", bestL)

	trainStaticResult = pd.DataFrame()
	testStaticResult = pd.DataFrame()

	# prepare to calculate confusion matrix
	# get the raw data
	for index, path in enumerate(allTrainFilePaths):
		trainFileItem, trainFileLabel = load_data_and_label(path, hasTime=False)
		gluedTrainFile = split_DF_in_parts(trainFileItem, parts=bestL, needConcat=True)
		gluedTrainFile.columns = gen_multiple_column_name(parts=bestL, hasTime=False)
		trainStaticResultItem = gen_statistic_result(gluedTrainFile, index + 1, hasTime=False)
		trainStaticResultItem["label"] = convert_label_2_num(trainFileLabel)
		trainStaticResult = trainStaticResult.append(trainStaticResultItem, sort=False)

	for index, path in enumerate(allTestFilePaths):
		testFileItem, testFileLabel = load_data_and_label(path, hasTime=False)
		gluedTestFile = split_DF_in_parts(testFileItem, parts=bestL, needConcat=True)
		gluedTestFile.columns = gen_multiple_column_name(parts=bestL, hasTime=False)
		testStaticResultItem = gen_statistic_result(gluedTestFile, index + 1, hasTime=False)
		testStaticResultItem["label"] = convert_label_2_num(testFileLabel)
		testStaticResult = testStaticResult.append(testStaticResultItem, sort=False)

	X_train = trainStaticResult[list(trainStaticResult.columns.values)[:-1]]
	y_train = trainStaticResult['label']

	X_test = testStaticResult[list(testStaticResult.columns.values)[:-1]]
	y_test = testStaticResult['label']

	params = {}
	classifier = GridSearchCV(MultinomialNB(), cv=StratifiedKFold(5), param_grid=params).fit(X_train, y_train)
	y_predict = classifier.predict(X_test)
	# Confusion Matrix
	# Returns the distance of each sample from the decision boundary for each class
	crosstab = pd.crosstab(y_test, y_predict, rownames=['True'],
						   colnames=['Predicted'], margins=True)
	print("-----------\"Confusion Matrix\"-------------")
	print(crosstab)
