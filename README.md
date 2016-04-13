# kmeans.py
kmeans 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

train = pd.read_csv(r'C:\Users\kent\Desktop\label.csv')
RFM   = train.ix[:,('R','F','M')]

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(RFM)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.get_params(RFM)


fig = plt.figure(1,figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(RFM)
labels = k_means.labels_
ax.scatter(RFM.ix[:,0], RFM.ix[:,1], RFM.ix[:,2], c=labels.astype(np.float))
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Rencntly')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monatary')
plt.show()
