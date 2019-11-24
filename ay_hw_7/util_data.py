#
import string

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '10/29/2019 10:14 AM'

import os
from ._global import SPLASH


def gen_corpus(file_pathes, output_file_name="CORPIS.txt"):
	dataFiles = os.listdir(file_pathes)
	resultFile = open(output_file_name, 'w')
	for fileItem in dataFiles:
		with open(file_pathes + SPLASH + fileItem, 'r', encoding='ascii', errors='ignore') as fileDataStream:
			for line in fileDataStream:
				resultFile.write(line)

	resultFile.close()
	return resultFile.name


def gen_chars_set(orginalText, ignore_case=False, remove_punctuation=False):
	trimedText = None
	charSet = set()
	if ignore_case:
		trimedText = orginalText = orginalText.lower()
	if remove_punctuation:
		charSet = set(orginalText).difference(set(string.punctuation))
		trimedText = orginalText.translate(str.maketrans('', '', string.punctuation))
	return trimedText, charSet
