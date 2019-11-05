# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/9/2019 6:48 PM'

import os

def get_all_datasets_path(rootPath):
	filePaths = []
	for path, directories, files in os.walk(rootPath):
		for file in files:
			filePaths.append(os.path.join(path, file))
	return filePaths

if __name__ == "__main__":
	allFilePaths = get_all_datasets_path(".\\assets")
	print(allFilePaths)