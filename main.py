import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output


def random_centroids(data, k):
    return data.sample(n=k).T


def get_labels(data, centroids):
    from scipy.spatial.distance import cdist

    distances = cdist(data.values, centroids.T.values)
    return distances.argmin(axis=1)


def new_centroids(data, labels, k):
    new_c = []
    for i in range(k):
        cluster_points = data[labels == i]
        new_c.append(cluster_points.mean())
    return pd.DataFrame(new_c).T


def plot_clusters(data, labels, centroids, iteration, save_dir="cluster_images"):
    os.makedirs(save_dir, exist_ok=True)

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)

    clear_output(wait=True)

    plt.figure(figsize=(8, 6))
    plt.title(f"Iteration {iteration}")
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.scatter(
        x=centroids_2d[:, 0],
        y=centroids_2d[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroids",
    )
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"iteration_{iteration:03d}.png"))
    plt.close()


from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

max_iterations = 100
centroid_count = 3

centroids = random_centroids(data, centroid_count)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, centroid_count)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1
