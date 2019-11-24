#


__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/3/2019 9:26 AM'

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data, train_test_split_by_size


def RidgeRegressionCV():
	return Pipeline([
		('std_sclaer', StandardScaler()),
		('ridge_reg', RidgeCV(cv=10))
	])


if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	X_train, X_test, y_train, y_test = train_test_split_by_size(X_data, y_data, train_size=1495, random_state=2333)
	ridgeCV = RidgeRegressionCV().fit(X_train, y_train)
	y_predict = ridgeCV.predict(X_test)
	print("-----------\"Mean Square Error\"-------------")
	print("Mean Square Error : ", mean_squared_error(y_test, y_predict))
	print("-----------\"Score\"-------------")
	print("Score : ", ridgeCV.score(X_test, y_test))
