# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 10:14 AM'
import numpy as np


def load_data(file_path):
	"""read records from csv file"""
	row_data = np.genfromtxt(file_path, dtype=None, delimiter=',', encoding='utf-8')[1:]
	X = np.array(row_data[:, :-1], dtype=float)
	y = np.array(row_data[:, row_data.shape[1] - 1], dtype=float)
	return X, y
