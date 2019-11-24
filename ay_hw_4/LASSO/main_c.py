# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/2/2019 9:25 PM'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	data = pd.concat([X_data, y_data], axis=1)
	correlation = data.corr()
	fig = plt.figure(figsize=(20, 15))
	sns.heatmap(correlation, vmin=-1, vmax=1, cmap=sns.color_palette("Blues"))
	plt.show()
