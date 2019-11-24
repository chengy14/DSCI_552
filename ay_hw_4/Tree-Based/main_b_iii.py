# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/4/2019 5:30 PM'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ay_hw_4._global import ROOT_PATH, APS_TRAIN, APS_FULL_COLUMNS
from ay_hw_4.util_data import load_data, to_binary_numeric

if __name__ == "__main__":
	pd.set_option('display.max_columns', 100)
	X_data, y_data = load_data(ROOT_PATH + APS_TRAIN, skip_first_row=21, y_column_index=0,
							   assignedColumnNames=APS_FULL_COLUMNS, missingSymbol='na', needImpute=True, dropOrNot=True)
	y_data = to_binary_numeric(y_data, classNeg="neg")
	data = pd.concat([y_data, X_data], axis=1)
	correlation = data.corr()
	fig = plt.figure(figsize=(20, 15))
	sns.heatmap(correlation, vmin=-1, vmax=1, cmap=sns.color_palette("Blues"))
	plt.show()

# 把dropOrNot打开 将报错， 因为数据中有10列存在NaN