#
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/3/2019 10:45 PM'
import warnings

import matplotlib.pyplot
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data, train_test_split_by_size

if __name__ == "__main__":
	warnings.simplefilter(action='ignore', category=FutureWarning)
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	X_train, X_test, y_train, y_test = train_test_split_by_size(X_data, y_data, train_size=1495, random_state=2333)
	d_train = xgb.DMatrix(X_train, label=y_train)
	d_test = xgb.DMatrix(X_test, label=y_test)

	xgb_clf = xgb.XGBRegressor(n_estimators=100, max_depth=6, objective="reg:squarederror", silent=False)

	parameters = {'reg_alpha': [1e-5, 1e-4, 1e-3, 0.01, 0.1]}
	grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10, n_jobs=-1)

	grid_search.fit(X_train, y_train)
	print("Best parameters alpha :", grid_search.best_params_)

	xgb.plot_tree(grid_search.best_estimator_, num_trees=1)
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(24, 6)
	matplotlib.pyplot.show()
