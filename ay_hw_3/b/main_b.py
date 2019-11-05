# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/24/2019 11:30 AM'

from ay_hw_3.util_data import load_multiple_files, get_all_datasets_path
from ay_hw_3.util_generate import gen_test_data_file_paths, gen_train_data_file_paths
from ay_hw_3._global import ROOT_PATH

'''(b) Keep datasets 1 and 2 in folders bending1 and bending 2, 
as well as datasets 1, 2, and 3 in other folders as test data 
and other datasets as train data.'''

if __name__ == "__main__":
	# find all the data files path in 'assets' dir
	allFilePaths = get_all_datasets_path(ROOT_PATH)
	print(allFilePaths)
	# based on what the pdf said, we need to use some specific file as test data
	testDataFileNames = gen_test_data_file_paths(ROOT_PATH)
	trainDataFileNames = gen_train_data_file_paths()

	testData = load_multiple_files(testDataFileNames)
	trainData = load_multiple_files(trainDataFileNames)
	print("Train Data [:3]: \n", trainData[:3])
	print("Train Data Shape: ", trainData.to_numpy().shape)
	print("Train Data [:3]: \n", testData[:3])
	print("Test Data Shape: ", testData.to_numpy().shape)
