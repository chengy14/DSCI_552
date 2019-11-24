# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/2/2019 10:50 PM'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt, floor

from ay_hw_4._global import ROOT_PATH, CRIME, CRIME_REMAIN_COLUMNS
from ay_hw_4.util_data import load_data

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	data = pd.concat([X_data, y_data], axis=1)
	cvFormula = lambda x: np.std(x) / np.mean(x)
	cvResult = np.apply_along_axis(cvFormula, axis=0, arr=data.to_numpy())

	first_sqrt_128 = floor(sqrt(128))
	cvResultIndexes = np.argsort(-cvResult)[:first_sqrt_128]
	highestCVFeatureNames = np.array(CRIME_REMAIN_COLUMNS)[cvResultIndexes]
	renderData = X_data[highestCVFeatureNames]

	fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))

	index = 0
	for a in range(0, 4):
		for b in range(0, 3):
			if index < 11:
				ax[a, b].boxplot(renderData.loc[:, highestCVFeatureNames[index]], whis=3, vert=False)
				ax[a, b].grid(True)
				ax[a, b].title.set_text(highestCVFeatureNames[index])
				b, index = b + 1, index + 1

	ax[3, 2].remove()
	plt.show()
