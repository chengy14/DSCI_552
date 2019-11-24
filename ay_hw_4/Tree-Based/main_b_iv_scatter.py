# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/4/2019 4:52 PM'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt, floor

from ay_hw_4._global import ROOT_PATH, APS_TRAIN, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data, to_binary_numeric

if __name__ == "__main__":
	pd.set_option('display.max_columns', 100)
	X_data, y_data = load_data(ROOT_PATH + APS_TRAIN, skip_first_row=21, y_column_index=0, missingSymbol='na',
							   assignedColumnNames=APS_FULL_COLUMNS, needImpute=True, dropOrNot=True)
	y_data = to_binary_numeric(y_data, classNeg="neg")
	data = pd.concat([y_data, X_data], axis=1)
	cvFormula = lambda x: np.std(x) / np.mean(x)
	cvResult = np.apply_along_axis(cvFormula, axis=0, arr=data.to_numpy())

	first_sqrt_170 = floor(sqrt(170))
	cvResultIndexes = np.argsort(-cvResult)[:first_sqrt_170]
	highestCVFeatureNames = np.array(list(data.columns.values))[cvResultIndexes]
	renderData = X_data[highestCVFeatureNames]

	sns.pairplot(renderData, plot_kws=dict(s=80, edgecolor="white", linewidth=2.5, alpha=0.6))
	plt.show()
# 把dropOrNot打开 将报错， 因为数据中有10列存在NaN
