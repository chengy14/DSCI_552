# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/26/2019 9:40 AM'

import sys
import warnings

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts,is_bending
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_label, gen_multiple_column_name
from ay_hw_3.util_statistic import gen_statistic_result

if __name__ == "__main__":

	if not sys.warnoptions:
		warnings.simplefilter("ignore")

	allTrainFilePaths = gen_train_data_file_paths()
	# based on what the pdf said, we need to use all training data
	trainStaticResult = pd.DataFrame()
	rfeSources = {}
	significantVars = dict()
	for parts in range(1, 5):
		for index, path in enumerate(allTrainFilePaths):
			fileItem, fileLabel = load_data_and_label(path, hasTime=False)
			gluedFile = split_DF_in_parts(fileItem, parts=parts, needConcat=True)

			gluedFile.columns = gen_multiple_column_name(parts=parts, hasTime=False)
			staticResultItem = gen_statistic_result(gluedFile, index + 1, hasTime=False)
			staticResultItem["label"] = is_bending(fileLabel)
			trainStaticResult = trainStaticResult.append(staticResultItem, sort=False)

		logitModel = LogisticRegression()
		recursiveFeatureEliminationObj = RFECV(estimator=logitModel, cv=StratifiedKFold(5), scoring='accuracy')
		trainStatColumns = list(trainStaticResult.columns.values)
		recursiveFeatureEliminationObj.fit(trainStaticResult[trainStatColumns[:-1]],
										   trainStaticResult['label'])

		labels = gen_multiple_label(parts)
		rfeSources[parts] = recursiveFeatureEliminationObj.grid_scores_[
			recursiveFeatureEliminationObj.n_features_ - 1]
		tempList = list()
		for index, value in enumerate(labels):
			if recursiveFeatureEliminationObj.support_[index] == True:
				tempList.append(value)

		significantVars[parts] = tempList
		# allocate a new space for trainStaticResult
		trainStaticResult = pd.DataFrame()

	bestL = [key for maxValue in [max(rfeSources.values())]
			 for key, val in rfeSources.items() if val == maxValue][0]
	print("Best l : ", bestL)
	selectedFeatures = significantVars[bestL]
	print("selectedFeatures : ", selectedFeatures)
