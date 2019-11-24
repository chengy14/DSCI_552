# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/2/2019 10:50 PM'
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

	sns.pairplot(renderData, plot_kws=dict(s=80, edgecolor="white", linewidth=2.5, alpha=0.6))
	plt.show()
