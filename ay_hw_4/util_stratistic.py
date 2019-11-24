# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/4/2019 10:13 PM'

import numpy as np

from ay_hw_4.util_data import to_binary_numeric


def count_neg_and_pos(y_data):
	y_value = to_binary_numeric(y_data)
	num_neg = np.count_nonzero(y_value)
	return len(y_value) - num_neg, num_neg
