#
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 7:04 PM'

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts, convert_label_2_num
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_column_name, gen_test_data_file_paths
from ay_hw_3.util_statistic import gen_statistic_result
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from warnings import simplefilter
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

	simplefilter(action='ignore', category=FutureWarning)
	allTrainFilePaths = gen_train_data_file_paths()
	allTestFilePaths = gen_test_data_file_paths()
	# based on what the pdf said, we need to use all training data

	trainStaticResult = pd.DataFrame()
	testStaticResult = pd.DataFrame()
	logitCVTestErrorRateList = list()

	for parts in range(1, 4):
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

		logitModel = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear").fit(X_train, y_train)
		logitCVTestErrorRateList.append(1 - logitModel.score(X_test, y_test))

		# allocate a new space for statisticResult
		trainStaticResult = pd.DataFrame()
		testStaticResult = pd.DataFrame()

	print("-----------\"Test Error Rate\"-------------")
	print("Test Error Rate : ", logitCVTestErrorRateList)
	# # find the best l
	bestL = logitCVTestErrorRateList.index(min(logitCVTestErrorRateList)) + 1
	# bestL = 1
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

	# build an l1 penalized multinomial regression model to classify
	# binary the multiple classes
	bin_y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5])
	bin_y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])

	classifier = OneVsRestClassifier(
		LogisticRegressionCV(cv=StratifiedKFold(5), penalty="l1", solver="liblinear")
	).fit(X_train, bin_y_train)

	# Confusion Matrix
	# Returns the distance of each sample from the decision boundary for each class
	y_score = classifier.fit(X_train, bin_y_train).decision_function(X_test)
	y_predict = classifier.predict(X_test)
	crosstab = pd.crosstab(bin_y_test.argmax(axis=1), y_predict.argmax(axis=1), rownames=['True'],
						   colnames=['Predicted'], margins=True)
	print("-----------\"Confusion Matrix\"-------------")
	print(crosstab)

	# Compute ROC curve and AUC for each class
	falsePositiveRate = dict()
	truePositiveRate = dict()
	areas = dict()
	for i in range(bin_y_train.shape[1]):
		falsePositiveRate[i], truePositiveRate[i], _ = roc_curve(bin_y_test[:, i], y_score[:, i])
		areas[i] = auc(falsePositiveRate[i], truePositiveRate[i])

	colors = ['#16a085', '#27ae60', '#3498db', '#f1c40f', '#e74c3c', '#8e44ad']
	for index in range(bin_y_train.shape[1]):
		plt.plot(falsePositiveRate[index], truePositiveRate[index], color=colors[index],
				 label='ROC (class = {}) auc = {}'.format(index, str(areas[index])), linewidth=2, linestyle='-')

	plt.plot([0, 1], [0, 1], linestyle='-.', linewidth=5, alpha=0.5, color='yellow', label='diagonal line')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC & AUC about multiple classes')
	plt.legend()
	plt.show()
