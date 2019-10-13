# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/11/2019 9:13 AM'

import pandas as pd

'''
iii. What are the mean, the median, range, first and third quartiles, and interquartile ranges of each of the variables 
in the dataset? Summarize them in a table.
'''

if __name__ == "__main__":
	df = pd.read_csv('../assets/data.csv')
	print("---------------------------------------------------------"
		  "--------------------------------------------------------")
	result = pd.DataFrame({
		"Mean": df.mean(),
		"Median": df.median(),
		"Range(Min-Max)": list(zip(df.min().get_values(), df.max().get_values())),
		"First Quartiles": df.quantile(.25),
		"Third Quartiles": df.quantile(.75),
		"Interquartile Range": df.quantile(.75) - df.quantile(.25)
	})
	print(result.to_string())
	print("---------------------------------------------------------"
		  "--------------------------------------------------------")
