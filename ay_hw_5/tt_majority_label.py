# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/1/2019 5:58 PM'
from statistics import mode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# clustering dataset
# determine k using elbow method
# k means determine k
if __name__ == "__main__":
	df = pd.read_csv('./assets/Frogs_MFCCs.csv')
	K = range(2, 15)
	ks = []
	dist = []
	blabels = []
	hamming_losses = []

	for i in tqdm(range(50)):
		diss = []
		silh = []
		results = []
		X = df.iloc[:, :-4]
		for k in range(2, 15):
			kmeanModel = KMeans(n_clusters=k).fit(X)
			label = kmeanModel.labels_
			results.append(label)
			silh.append(silhouette_score(X, label))
			diss.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

		index = np.argmax(silh)
		best_k = index + 2
		ks.append(best_k)
		dist.append(diss)
		best_labels = results[index]
		blabels.append(best_labels)
		print('The best value of K is', best_k)

		labels_df = df[['Family', 'Genus', 'Species']].copy()
		labels_df['kmeans_label'] = best_labels
		majority_label = {}
		for l in range(best_k):
			cluster = labels_df[labels_df['kmeans_label'] == l]
			majority_triplet = {}
			for tl in ['Family', 'Genus', 'Species']:
				majority_triplet[tl] = cluster[tl].value_counts().idxmax()
			majority_label[l] = majority_triplet
		print("Majority labels:", majority_label)

		# compute hamming loss

		misclassifed = 0
		for l in range(best_k):
			cluster = labels_df[labels_df['kmeans_label'] == l]
			for tl in ['Family', 'Genus', 'Species']:
				misclassifed += sum(cluster[tl] != majority_label[l][tl])
		hamming_loss = misclassifed / (len(df) * 3)
		hamming_losses.append(hamming_loss)

	kk = np.argmax(mode(ks)) + 2
	print('The values of k is', mode(ks))
	print('Best Labels are', blabels[kk])
	print('Mean of Hamming Loss', np.mean(hamming_losses))
	print('Std of Hamming Loss', np.std(hamming_losses))

	distortions = []
	for i in range(len(dist)):
		av = 0
		for j in range(len(dist)):
			av += dist[j][i]
		distortions.append(av / len(dist))

	# Plot the elbow
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.show()
