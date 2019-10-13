# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/20/2019 11:22 PM'

import numpy as np
import math

if __name__ == "__main__":
	X_data = np.array([[0.0, 3, 0], [2, 0, 0], [0, 1, 3],
				  [0, 1, 2], [-1, 0, 1], [1, 1, 1]], dtype=float)
	X_zero = [0, 0, 0]
	distances = np.array([math.sqrt(np.sum(item - X_zero) ** 2) for item in X_data], dtype=float)

	y_data = np.array(['Red', 'Red', 'Red', 'Green', 'Green', 'Red'])

	print(distances)
