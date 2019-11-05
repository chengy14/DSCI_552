# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/25/2019 2:04 PM'

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts,is_bending
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_column_name
from ay_hw_3.util_statistic import gen_statistic_result
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":

	allTrainFilePaths = gen_train_data_file_paths()
	# based on what the pdf said, we need to use all training data
	trainStaticResult = pd.DataFrame()
	for index, path in enumerate(allTrainFilePaths):
		fileItem, fileLabel = load_data_and_label(path, hasTime=False)
		gluedFile = split_DF_in_parts(fileItem, parts=2, needConcat=True)
		gluedFile.columns = gen_multiple_column_name(hasTime=False)
		staticResultItem = gen_statistic_result(gluedFile, index + 1, hasTime=False)
		staticResultItem["label"] = is_bending(fileLabel)
		trainStaticResult = trainStaticResult.append(staticResultItem)

	features = ['min(1)', 'max(1)', 'mean(1)', 'min(2)', 'max(2)', 'mean(2)', 'min(12)', 'max(12)', 'mean(12)', 'label']
	subStatisticResult = trainStaticResult[features]
	# print(subStatisticResult[:3])
	sns.pairplot(subStatisticResult, hue="label", markers=["o", "+"])
	plt.show()
