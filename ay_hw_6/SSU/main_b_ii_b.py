#
from warnings import simplefilter

from tqdm import tqdm

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/5/2019 8:23 PM'

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support as score, roc_curve, auc, accuracy_score

from ay_hw_6._global import ROOT_PATH, WDBC_FILE_PATH, SPLASH, M
from ay_hw_6.util_data import load_data, train_test_split_by_class_and_ratio
from ay_hw_6.util_plot import plot_roc


def fit_semi_learning_model(X_train, y_train, X_test, clf):
	X_test_shape = X_test.shape[0]
	for _ in range(X_test_shape):
		absDist = np.absolute(bestClf.decision_function(X_test))
		indexOfMaxDist = np.argmax(absDist)
		farthestDataPoint = np.array(X_test[indexOfMaxDist]).reshape(1, 30)
		predictedResult = clf.predict(farthestDataPoint)

		X_test = np.delete(X_test, indexOfMaxDist, axis=0)
		X_train = np.append(X_train, farthestDataPoint, axis=0)
		y_train = np.append(y_train, predictedResult, axis=0)

		clf.fit(X_train, y_train)

	return clf


if __name__ == "__main__":
	simplefilter(action='ignore', category=ConvergenceWarning)
	X_data, y_data = load_data(ROOT_PATH + SPLASH + WDBC_FILE_PATH)
	X_data = Normalizer().fit(X_data).transform(X_data)
	labelEncoder = LabelEncoder().fit(['B', 'M'])
	y_data = labelEncoder.transform(y_data)
	X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																		   random_state=2333, pos_class=0, neg_class=1)


	parameters = {"C": np.linspace(1, 5, 10)}

	trainAccuracyList = list()
	trainPrecisionList = list()
	trainRecallRateList = list()
	trainF_scoreList = list()
	trainAUCList = list()

	testAccuracyList = list()
	testPrecisionList = list()
	testRecallRateList = list()
	testF_scoreList = list()
	testAUCList = list()
	trainCrosstabList = dict()
	testCrosstabList = dict()
	y_trainList = dict()
	y_testList = dict()
	y_train_predictList = dict()
	y_test_predictList = dict()
	y_train_trueList = dict()
	for i in tqdm(range(2)):
		labeled_X, unlabeled_X, labeled_y, expert = train_test_split_by_class_and_ratio(X_train, y_train, test_size=0.5,
																						random_state=i, pos_class=0,
																						neg_class=1)
		gridCV = GridSearchCV(LinearSVC(penalty="l1", dual=False, max_iter=5000), parameters, cv=5, n_jobs=-1)
		gridCV.fit(labeled_X, labeled_y)
		bestClf = gridCV.best_estimator_

		bestClf = fit_semi_learning_model(labeled_X.copy(), labeled_y.copy(), unlabeled_X.copy(), bestClf)

		y_train_predict = bestClf.predict(labeled_X)
		y_test_predict = bestClf.predict(X_test)

		y_train_predictList[i] = y_train_predict
		y_test_predictList[i] = y_test_predict
		y_train_trueList[i] = labeled_y

		trainCrosstab = pd.crosstab(labeled_y, y_train_predict, rownames=['True'], colnames=['Predicted'], margins=True)
		trainCrosstabList[i] = trainCrosstab
		testCrosstab = pd.crosstab(y_test, y_test_predict, rownames=['True'], colnames=['Predicted'], margins=True)
		testCrosstabList[i] = testCrosstab

		trainAccuracyList.append(accuracy_score(labeled_y, y_train_predict))
		testAccuracyList.append(accuracy_score(y_test, y_test_predict))

		trainPrecision, trainRecall, trainF_score, _ = score(labeled_y, y_train_predict, average='binary', pos_label=0)
		trainPrecisionList.append(trainPrecision)
		trainRecallRateList.append(trainRecall)
		trainF_scoreList.append(trainF_score)

		testPrecision, testRecall, testF_score, _ = score(y_test, y_test_predict, average='binary', pos_label=0)
		testPrecisionList.append(testPrecision)
		testRecallRateList.append(testRecall)
		testF_scoreList.append(testF_score)

		falsePositiveRate, truePositiveRate, _ = roc_curve(labeled_y, y_train_predict)
		trainAUC = auc(falsePositiveRate, truePositiveRate)
		trainAUCList.append(trainAUC)

		falsePositiveRate, truePositiveRate, _ = roc_curve(y_test, y_test_predict)
		testAUC = auc(falsePositiveRate, truePositiveRate)
		testAUCList.append(testAUC)

	indexOfbBestPrecision = np.argmax(trainPrecisionList)
	print("-----------\"Training Confusion Matrix\"-------------")
	print(trainCrosstabList[indexOfbBestPrecision])
	print("-----------\"Training AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(trainAccuracyList))
	print("AVG Precision: ", np.average(trainPrecisionList))
	print("AVG Recall Rate: ", np.average(trainRecallRateList))
	print("AVG F1 -Score: ", np.average(trainF_scoreList))
	print("AVG AUC: ", np.average(trainAUCList))
	semiTrainDF = pd.DataFrame(data={'Algorithm': ['Semi-Supervised Train']})
	semiTrainDF['AVG Accuracy'] = np.average(trainAccuracyList)
	semiTrainDF['AVG Precision'] = np.average(trainPrecisionList)
	semiTrainDF['AVG Recall'] = np.average(trainRecallRateList)
	semiTrainDF['AVG F Score'] = np.average(trainF_scoreList)
	semiTrainDF['AVG AUC'] = np.average(trainAUCList)
	# infoDF = infoDF.append(semiTrainDF)
	plot_roc(y_train_trueList[indexOfbBestPrecision], y_train_predictList[indexOfbBestPrecision], title="Training ROC")

	indexOfbBestPrecision = np.argmax(testPrecisionList)
	print("-----------\"Testing Confusion Matrix\"-------------")
	print(testCrosstabList[indexOfbBestPrecision])
	print("-----------\"Testing AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(testAccuracyList))
	print("AVG Precision: ", np.average(testPrecisionList))
	print("AVG Recall Rate: ", np.average(testRecallRateList))
	print("AVG F1 -Score: ", np.average(testF_scoreList))
	print("AVG AUC: ", np.average(testAUCList))
	semiTestDF = pd.DataFrame(data={'Algorithm': ['Semi-Supervised Test']})
	semiTestDF['AVG Accuracy'] = np.average(testAccuracyList)
	semiTestDF['AVG Precision'] = np.average(testPrecisionList)
	semiTestDF['AVG Recall'] = np.average(testRecallRateList)
	semiTestDF['AVG F Score'] = np.average(testF_scoreList)
	semiTestDF['AVG AUC'] = np.average(testAUCList)
	# infoDF = infoDF.append(semiTestDF)
	plot_roc(y_test, y_test_predictList[indexOfbBestPrecision], title="Testing ROC")
