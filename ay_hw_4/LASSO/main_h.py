#


__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/3/2019 10:09 AM'

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data, train_test_split_by_size


def LassoRegressionCV():
	return Pipeline([
		('std_sclaer', StandardScaler()),
		('lasso_cv', LassoCV(cv=10, random_state=233))
	])


if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	X_train, X_test, y_train, y_test = train_test_split_by_size(X_data, y_data, train_size=1495, random_state=2333)
	lassoNormalizedReg = LassoCV(cv=10, random_state=233).fit(X_train, y_train)
	y_predict = lassoNormalizedReg.predict(X_test)
	print("-----------\"Mean Square Error(Normalized)\"-------------")
	print("Mean Square Error : ", mean_squared_error(y_test, y_predict))
	print("-----------\"Score\"-------------")
	print("Score : ", lassoNormalizedReg.score(X_test, y_test))

	lassoStandardizedReg = LassoRegressionCV().fit(X_train, y_train)
	y_predict = lassoStandardizedReg.predict(X_test)
	print("-----------\"Mean Square Error(Standardized)\"-------------")
	print("Mean Square Error : ", mean_squared_error(y_test, y_predict))
	print("-----------\"Score\"-------------")
	print("Score : ", lassoStandardizedReg.score(X_test, y_test))