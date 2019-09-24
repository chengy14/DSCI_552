# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '8/28/2019 9:32 PM'
import matplotlib.pyplot as plt
from ay_hw_1.util import load_data


'''
ii. Make boxplots for each of the independent variables. Use color to show
Classes 0 and 1 (see ISLR p. 129)
'''
if __name__ == "__main__":
	# load data from csv file
	X_train, y_train = load_data('../assets/data.csv')
	labels = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
			  'degree_spondylolisthesis']
	results = ['Normal', 'Abnormal']
	colors = ['#16a085', '#27ae60', '#3498db', '#f1c40f', '#e74c3c', '#8e44ad']
	fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(15, 12))

	normal_data = []
	abnormal_data = []
	all_data = []

	normal_square = dict(markerfacecolor='#ffffff', marker='s')

	for index in range(0, 6):
		temp = []
		temp.append(X_train[y_train == 0, index])
		temp.append(X_train[y_train == 1, index])
		ax[index, 0].boxplot(temp, labels=results, flierprops=dict(markerfacecolor=colors[index], marker='s'),
							 whis=0.85, notch=True,
							 medianprops={'color': colors[index]}, whiskerprops={'color': colors[index]},
							 capprops={'color': colors[index]}, vert=False)
		ax[index, 0].grid(True)
		ax[index, 0].title.set_text(labels[index] + " Symptom")
		ax[index, 1].remove()

	plt.show()
