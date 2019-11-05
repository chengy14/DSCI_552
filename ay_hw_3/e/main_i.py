# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 7:04 PM'

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts,is_bending
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_column_name
from ay_hw_3.util_statistic import gen_statistic_result
from sklearn.linear_model import LogisticRegressionCV
from warnings import simplefilter
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

	simplefilter(action='ignore', category=FutureWarning)
	allTrainFilePaths = gen_train_data_file_paths()
	# based on what the pdf said, we need to use all training data
	statisticResult = pd.DataFrame()
	logitCVTestScores = list()
	for parts in range(1, 21):
		for index, path in enumerate(allTrainFilePaths):
			fileItem, fileLabel = load_data_and_label(path, hasTime=False)
			gluedFile = split_DF_in_parts(fileItem, parts=parts, needConcat=True)
			gluedFile.columns = gen_multiple_column_name(parts=parts, hasTime=False)
			staticResultItem = gen_statistic_result(gluedFile, index + 1, hasTime=False)
			staticResultItem["label"] = is_bending(fileLabel)
			statisticResult = statisticResult.append(staticResultItem, sort=False)

		X_data = statisticResult[list(statisticResult.columns.values)[:-1]]
		y_data = statisticResult['label']
		logitModel = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear").fit(X_data, y_data)
		logitCVTestScores.append(logitModel.score(X_data, y_data))

		# allocate a new space for statisticResult
		statisticResult = pd.DataFrame()

	print("-----------\"Test Error Rate\"-------------")
	print("Test Error Rate : ", logitCVTestScores)
	# # find the best l
	bestL = logitCVTestScores.index(max(logitCVTestScores)) + 1
	# bestL = 1
	print("-----------\"Best L\"-------------")
	print("Best l : ", bestL)

	plt.plot(range(1, 21), logitCVTestScores, color='red', label='score')
	plt.vlines(bestL, 0, 1, linestyles='dotted', label="best l = " + str(bestL))
	plt.xlabel('The Number of Breaking Times')
	plt.ylabel('L1 LR Scores')
	plt.xlim(0, 21)
	plt.ylim(0.8, 1.05)
	plt.legend()
	plt.show()
