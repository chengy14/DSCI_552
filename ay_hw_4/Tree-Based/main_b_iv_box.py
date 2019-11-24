# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/4/2019 4:52 PM'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

	fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(25, 15))

	index = 0
	for a in range(0, 5):
		for b in range(0, 3):
			if index < 13:
				ax[a, b].boxplot(renderData.loc[:, highestCVFeatureNames[index]], whis=2, vert=False)
				ax[a, b].grid(True)
				ax[a, b].title.set_text(highestCVFeatureNames[index])
				b, index = b + 1, index + 1

	ax[4, 1].remove()
	ax[4, 2].remove()
	plt.show()
