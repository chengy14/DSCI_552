# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/28/2019 1:32 PM'
from ay_hw_1.util import load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''
i. Make scatterplots of the independent variables in the dataset. Use color to
show Classes 0 and 1.
'''
if __name__ == "__main__":
	# load data from csv file
	X_train, y_train = load_data('../assets/data.csv')


	fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(32, 24))
	labels = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
			  'degree_spondylolisthesis']
	result = ['Normal', 'Abnormal']


	for x_index in range(0, 6):
		for y_index in range(0, 6):
			if (x_index != y_index):
				ax[x_index, y_index].scatter(X_train[y_train == 0, x_index], X_train[y_train == 0, y_index],
											 color='#0984e3', label=result[0], alpha=0.6, edgecolors='none')
				ax[x_index, y_index].scatter(X_train[y_train == 1, x_index], X_train[y_train == 1, y_index],
											 color='#ff7675', label=result[1], alpha=0.6, edgecolors='none')
				ax[x_index, y_index].set_xlabel(labels[x_index])
				ax[x_index, y_index].set_ylabel(labels[y_index])
				ax[x_index, y_index].grid(True)

			y_index += 1
		x_index += 1


	fig.tight_layout()
	plt.show()
