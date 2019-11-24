# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/2/2019 10:36 PM'

import pandas as pd
import numpy as np

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	data = pd.concat([X_data, y_data], axis=1)
	cvFormula = lambda x: np.std(x) / np.mean(x)
	cvResult = np.apply_along_axis(cvFormula, axis=0, arr=data.to_numpy())
	print("The total {} features CV are: (first 20 rows)\n {}".format(len(cvResult), cvResult))
