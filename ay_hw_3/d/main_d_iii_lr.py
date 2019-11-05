# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/26/2019 9:40 AM'

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts,is_bending
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_label
from ay_hw_3.util_statistic import gen_statistic_result
import warnings
import statsmodels.api as sm
import pandas as pd
import sys

if __name__ == "__main__":

	if not sys.warnoptions:
		warnings.simplefilter("ignore")

	allTrainFilePaths = gen_train_data_file_paths()
	# based on what the pdf said, we need to use all training data
	trainStaticResult = pd.DataFrame()
	for parts in range(1, 21):
		for index, path in enumerate(allTrainFilePaths):
			fileItem, fileLabel = load_data_and_label(path, hasTime=False)
			splitedDFs = split_DF_in_parts(fileItem, parts=parts, needConcat=False)
			statisticResultTemp = pd.DataFrame()
			for DFItem in splitedDFs:
				staticResultTempItem = gen_statistic_result(DFItem, index + 1, hasTime=False)
				statisticResultTemp = statisticResultTemp.append(staticResultTempItem, sort=False)

			statisticResultTemp["label"] = is_bending(fileLabel)
			trainStaticResult = trainStaticResult.append(statisticResultTemp, sort=False)

		logitModel = sm.Logit(trainStaticResult['label'],
							  trainStaticResult[gen_multiple_label(parts=1)])
		logitModelResults = logitModel.fit(method="bfgs",disp=0)
		# ['median(1)'] ['max(5)']
		significantVars = \
			[key for key, p_value in logitModelResults.pvalues.items() if p_value <= 0.05]
		if len(significantVars) > 0:
			print("When split all training data sets in {} times, "
				  "I got significant variables : ".format(parts), end=" ")
			print(' '.join(significantVars))
		significantVars = []
		# allocate a new space for statisticResult
		trainStaticResult = pd.DataFrame()
