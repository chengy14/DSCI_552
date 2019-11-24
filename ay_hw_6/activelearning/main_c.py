# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/8/2019 9:01 PM'

import numpy as np

if __name__ == "__main__":
	data = np.arange(6).reshape((3, 2))
	result = np.average(data, axis=0)
	print(result)
	# avgPassiveErrorList = np.average(np.array(passiveOverallErrorList), axis=0)
	# avgActiveErrorList = np.average(np.array(activeOverallErrorList), axis=0)
	# plt.plot(np.arange(10, 901, 10), avgPassiveErrorList, marker='', linewidth=1, label="passive")
	# plt.plot(np.arange(10, 901, 10), avgActiveErrorList, marker='', linewidth=1, label="active")
	# plt.xlabel('The Size Of Training Instance')
	# plt.ylabel('Avg Test Error')
	# plt.ylim([0, 0.15])
	# plt.title("Learning Curve")
	# plt.legend()
	# plt.show()