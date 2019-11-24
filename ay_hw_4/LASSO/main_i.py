# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/3/2019 9:54 PM'

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import numpy as np
from sklearn.metrics import mean_squared_error
from ay_hw_4._global import ROOT_PATH, CRIME
from ay_hw_4.util_data import load_data, train_test_split_by_size

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + CRIME, skip_first_column=5, y_column_index=-1, needImpute=True)
	X_train, X_test, y_train, y_test = train_test_split_by_size(X_data, y_data, train_size=1495, random_state=2333)
	linReg = LinearRegression()
	mse = list()
	score = -1 * cross_val_score(linReg, np.ones((len(X_train), 1)), y_train, cv=10,
								 scoring='neg_mean_squared_error').mean()
	mse.append(score)
	for m in range(1, X_data.shape[1]):
		pca = PCA(n_components=m)
		X_train_reduced = pca.fit_transform(scale(X_train))
		mse.append(-1 * cross_val_score(linReg, X_train_reduced, y_train, cv=10,
										scoring='neg_mean_squared_error').mean())

	bestM = np.argsort(mse)[0]
	print("-----------\"Best M\"-------------")
	print("Best M : ", bestM)
	pca = PCA(n_components=bestM)
	X_train_reduced = pca.fit_transform(scale(X_train))
	X_test_reduced = pca.fit_transform(scale(X_test))
	bestReg = LinearRegression().fit(X_train_reduced, y_train)
	y_predict = bestReg.predict(X_test_reduced)
	print("-----------\"Mean Square Error\"-------------")
	print("Mean Square Error : ", mean_squared_error(y_test, y_predict))
