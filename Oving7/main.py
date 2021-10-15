import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Exercise 7

# * Using the UCI Mushroom dataset, use k-means and a suitable cluster
# evaluation metric to determine the optimal number of clusters in the
# dataset. Note that this may not necessarily be two (edible versus
# not-edible).
#
# * Plot this metric while increasing the number of clusters,
# e.g., $k=2..30$ (see [here](
# http://scikit-learn.org/stable/auto_examples/cluster
# /plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot
# -adjusted-for-chance-measures-py) for an example).
#
# * Visualise the data
# using the number of clusters and a suitable projection or low-dimensional
# embedding.

data = pd.read_csv('./agaricus-lepiota.csv')
data.pop('edibility')
dummies = pd.get_dummies(data)
data.head()

pca = PCA()
pca.fit(dummies)
amount = len(pca.explained_variance_)
plt.figure()
plt.plot(range(0, amount), pca.explained_variance_ratio_.cumsum(),
         linestyle='--', linewidth=2)
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

pca = PCA(n_components=20)
transform_pca = pca.fit_transform(dummies)
results = []
for i in range(2, 31):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++')
    kmeans_pca.fit(transform_pca)
    results.append(metrics.silhouette_score(transform_pca, kmeans_pca.labels_,
                                            metric='euclidean'))
plt.figure()
plt.plot(range(2, 31), results, marker="o")
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()

n = 7
results = np.array(results)
idx = np.argpartition(results, -n)[-n:]
indices = idx[np.argsort((-results)[idx])] + 2
print(indices)

k = indices[0]
labels = KMeans(n_clusters=k).fit_predict(transform_pca)
pca = PCA(n_components=3)
x_plot = np.array(pca.fit_transform(dummies))
fig = plt.figure(figsize=(10, 10))
fig.suptitle(f"K = {k} ")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2], c=labels, s=50,
           cmap='viridis')

k = indices[1]
labels = KMeans(n_clusters=k).fit_predict(transform_pca)
pca = PCA(n_components=3)
x_plot = np.array(pca.fit_transform(dummies))
fig = plt.figure(figsize=(10, 10))
fig.suptitle(f"K = {k} ")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2], c=labels, s=50,
           cmap='viridis')
plt.show()
