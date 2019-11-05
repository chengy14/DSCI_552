# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/27/2019 11:13 PM'
from ay_hw_3.util_data import convert_label_2_num, load_data_and_label
from  ay_hw_3.util_generate import  gen_test_data_file_paths
from ay_hw_3.util_statistic import gen_statistic_result
from ay_hw_3._global import FULL_COLUMNS, ROOT_PATH

if __name__ == "__main__":
	print(convert_label_2_num("bending1"))
	print(convert_label_2_num("bending2"))

	print(convert_label_2_num("cycling"))
	print(convert_label_2_num("sitting"))
	print(convert_label_2_num("walking"))
	print(type(convert_label_2_num("lying")))

	# print(gen_test_data_file_paths('.\\assets'))

	dataframe, label = load_data_and_label('.\\assets\\cycling\\dataset2.csv')
	staticResultItem = gen_statistic_result(dataframe, 1)
	print(staticResultItem.to_string())