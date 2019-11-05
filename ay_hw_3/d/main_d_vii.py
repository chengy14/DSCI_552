# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 5:41 PM'

from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

from ay_hw_3.util_data import load_data_and_label, split_DF_in_parts, is_bending
from ay_hw_3.util_generate import gen_train_data_file_paths, gen_multiple_column_name
from ay_hw_3.util_statistic import gen_statistic_result

if __name__ == "__main__":
	simplefilter(action='ignore', category=FutureWarning)
	# the best l I got is 3
	bestL = 3
	selectedFeatures = ['min(5)', '3rd quart(5)', '3rd quart(7)', 'max(18)', 'label']
	allTrainFilePaths = gen_train_data_file_paths()
	# based on what the pdf said, we need to use all training data
	statisticResult = pd.DataFrame()
	for index, path in enumerate(allTrainFilePaths):
		fileItem, fileLabel = load_data_and_label(path, hasTime=False)
		gluedFile = split_DF_in_parts(fileItem, parts=bestL, needConcat=True)
		gluedFile.columns = gen_multiple_column_name(parts=bestL, hasTime=False)
		staticResultItem = gen_statistic_result(gluedFile, index + 1, hasTime=False)
		staticResultItem["label"] = is_bending(fileLabel)
		statisticResult = statisticResult.append(staticResultItem, sort=False)

	bendingDF = statisticResult[selectedFeatures][statisticResult.label == 1]
	otherTypeDF = statisticResult[selectedFeatures][statisticResult.label == 0]

	print(bendingDF[:10])
	bootstrappedBendingDF = resample(bendingDF, replace=True, n_samples=60, random_state=2333)
	gluedBendingDF = pd.concat([otherTypeDF, bootstrappedBendingDF])

	X_data = gluedBendingDF[list(gluedBendingDF.columns.values)[:-1]]
	y_data = gluedBendingDF['label']

	logitModel = LogisticRegression()
	logitModel.fit(X_data, y_data)
	y_preidct = logitModel.predict(X_data)
	crosstab = pd.crosstab(y_data, y_preidct, rownames=['True'], colnames=['Predicted'], margins=True)
	print("-----------\"Confusion Matrix\"-------------")
	print(crosstab)
	falsePositiveRate, truePositiveRate, thresholds = roc_curve(y_data, y_preidct)
	# compute Area Under the Curve (AUC) using the trapezoidal rule
	area = auc(falsePositiveRate, truePositiveRate)
	plt.plot(falsePositiveRate, truePositiveRate, color='red', label='ROC' + str(area))
	plt.plot([0, 1], [0, 1], linestyle='dotted')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC & AUC')
	plt.legend()
	plt.show()
	print("-----------\"AUC value\"-------------")
	print("AUC : ", area)
