# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/29/2019 2:30 PM'


def accuracy_score(y_true, y_predict):
	'''compare every single item in each array'''
	assert y_true.shape[0] == y_predict.shape[0], \
		"the shape of y_true need to be identical with y_predict"
	return sum(y_true == y_predict) / len(y_true)
