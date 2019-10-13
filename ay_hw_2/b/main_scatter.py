# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/9/2019 8:57 PM'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
ii. Make pairwise scatterplots of all the varianbles in the data set including the
predictors (independent variables) with the dependent variable. Describe
your findings.
'''
if __name__ == "__main__":

	df = pd.read_csv('../assets/data.csv')
	sns.set_context("paper")
	grid = sns.pairplot(df, palette="Set2", diag_kind="kde", height=4, kind='scatter')
	plt.show()

