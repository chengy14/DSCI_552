#
import warnings

from tqdm import tqdm

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/5/2019 1:38 PM'

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn.svm import LinearSVC

from ay_hw_6._global import ROOT_PATH, WDBC_FILE_PATH, SPLASH, M
from ay_hw_6.util_data import load_data, train_test_split_by_class_and_ratio
from ay_hw_6.util_plot import plot_roc

warnings.filterwarnings("ignore")

if __name__ == "__main__":
	X_data, y_data = load_data(ROOT_PATH + SPLASH + WDBC_FILE_PATH)
	X_data = Normalizer().fit(X_data).transform(X_data)
	labelEncoder = LabelEncoder().fit(['B', 'M'])
	y_data = labelEncoder.transform(y_data)
	parameters = {"C": np.linspace(1, 8, 10)}
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
	for i in tqdm(range(M)):
		X_train, X_test, y_train, y_test = train_test_split_by_class_and_ratio(X_data, y_data, test_size=0.2,
																			   random_state=i, pos_class=0, neg_class=1)
		y_trainList[i] = y_train
		y_testList[i] = y_test

		linSVC = LinearSVC(penalty="l1", dual=False, max_iter=5000)
		gridCV = GridSearchCV(linSVC, parameters, cv=5, n_jobs=-1)
		gridCV.fit(X_train, y_train)

		bestClf = gridCV.best_estimator_
		y_train_predict = bestClf.predict(X_train)
		y_test_predict = bestClf.predict(X_test)

		y_train_predictList[i] = y_train_predict
		y_test_predictList[i] = y_test_predict

		trainCrosstab = pd.crosstab(y_train, y_train_predict, rownames=['True'], colnames=['Predicted'], margins=True)
		trainCrosstabList[i] = trainCrosstab
		testCrosstab = pd.crosstab(y_test, y_test_predict, rownames=['True'], colnames=['Predicted'], margins=True)
		testCrosstabList[i] = testCrosstab

		trainPrecision, trainRecall, trainF_score, _ = score(y_train, y_train_predict, average='binary', pos_label=0)
		trainPrecisionList.append(trainPrecision)
		trainRecallRateList.append(trainRecall)
		trainF_scoreList.append(trainF_score)

		testPrecision, testRecall, testF_score, _ = score(y_test, y_test_predict, average='binary', pos_label=0)
		testPrecisionList.append(testPrecision)
		testRecallRateList.append(testRecall)
		testF_scoreList.append(testF_score)

		falsePositiveRate, truePositiveRate, _ = roc_curve(y_train, y_train_predict)
		trainAUC = auc(falsePositiveRate, truePositiveRate)
		trainAUCList.append(trainAUC)

		falsePositiveRate, truePositiveRate, _ = roc_curve(y_test, y_test_predict)
		testAUC = auc(falsePositiveRate, truePositiveRate)
		testAUCList.append(testAUC)

		trainAccuracyList.append(accuracy_score(y_train, y_train_predict))
		testAccuracyList.append(accuracy_score(y_test, y_test_predict))

	indexOfbBestPrecision = np.argmax(trainPrecisionList)
	print("-----------\"Training Confusion Matrix\"-------------")
	print(trainCrosstabList[indexOfbBestPrecision])
	print("-----------\"Training AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(trainAccuracyList))
	print("AVG Precision: ", np.average(trainPrecisionList))
	print("AVG Recall Rate: ", np.average(trainRecallRateList))
	print("AVG F1 -Score: ", np.average(trainF_scoreList))
	print("AVG AUC: ", np.average(trainAUCList))
	supTrainDF = pd.DataFrame(data={'Algorithm': ['Supervised Train']})
	supTrainDF['AVG Accuracy'] = np.average(trainPrecisionList)
	supTrainDF['AVG Precision'] = np.average(trainPrecisionList)
	supTrainDF['AVG Recall'] = np.average(trainRecallRateList)
	supTrainDF['AVG F Score'] = np.average(trainF_scoreList)
	supTrainDF['AVG AUC'] = np.average(trainAUCList)
	infoDF = pd.DataFrame().append(supTrainDF)
	plot_roc(y_trainList[indexOfbBestPrecision], y_train_predictList[indexOfbBestPrecision], title="Training ROC")

	indexOfbBestPrecision = np.argmax(testPrecisionList)
	print("-----------\"Testing Confusion Matrix\"-------------")
	print(testCrosstabList[indexOfbBestPrecision])
	print("-----------\"Testing AVG Infos\"-------------")
	print("AVG Accuracy Score: ", np.average(testAccuracyList))
	print("AVG Precision: ", np.average(testPrecisionList))
	print("AVG Recall Rate: ", np.average(testRecallRateList))
	print("AVG F1 -Score: ", np.average(testF_scoreList))
	print("AVG AUC: ", np.average(testAUCList))
	supTestDF = pd.DataFrame(data={'Algorithm': ['Supervised Test']})
	supTestDF['AVG Accuracy'] = np.average(testAccuracyList)
	supTestDF['AVG Precision'] = np.average(testPrecisionList)
	supTestDF['AVG Recall'] = np.average(testRecallRateList)
	supTestDF['AVG F Score'] = np.average(testF_scoreList)
	supTestDF['AVG AUC'] = np.average(testAUCList)

	infoDF = infoDF.append(supTestDF)
	plot_roc(y_testList[indexOfbBestPrecision], y_test_predictList[indexOfbBestPrecision], title="Testing ROC")
