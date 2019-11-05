# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/25/2019 8:49 AM'

from ay_hw_3.util_data import load_data_and_label,is_bending
from ay_hw_3.util_generate import gen_train_data_file_paths
from ay_hw_3.util_statistic import gen_statistic_result

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
	# get all training data file paths
	allTrainFilePaths = gen_train_data_file_paths()

	trainStaticResult = pd.DataFrame()
	for index, path in enumerate(allTrainFilePaths):
		fileItem, fileLabel = load_data_and_label(path)
		staticResultItem = gen_statistic_result(fileItem, index + 1)
		staticResultItem["label"] = is_bending(fileLabel)
		trainStaticResult = trainStaticResult.append(staticResultItem)

	# ----------------same to the main_c_ii.py------------------
	features = ['min(1)', 'max(1)', 'mean(1)', 'min(2)', 'max(2)', 'mean(2)', 'min(6)', 'max(6)', 'mean(6)', 'label']
	subStatisticResult = trainStaticResult[features]
	# print(subStatisticResult.to_string())
	sns.pairplot(subStatisticResult, hue="label", markers=["o", "+"])
	plt.show()
