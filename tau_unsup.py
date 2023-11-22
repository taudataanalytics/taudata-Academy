# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 04:52:49 2018
Kumpulan Fungsi EMpirical Analysis Hard Clustering
Mata Kuliah Data Mining Prodi Statistika FMIPA UI
Semester Ganjil 2018
@author: Taufik Sutanto
"""

import numpy as np, time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)


def km_assumptions(n_samples= 1500, random_state = 170):
    # Author: Phil Roth <mr.phil.roth@gmail.com>
    # License: BSD 3 clause
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
    plt.figure(figsize=(12, 12))
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)# Incorrect number of clusters
    plt.subplot(221); plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")

    # Anisotropicly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

    plt.subplot(222); plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
    plt.title("Anisotropicly Distributed Blobs")

    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

    plt.subplot(223); plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
    plt.title("Unequal Variance")

    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

    plt.subplot(224); plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
    plt.title("Unevenly Sized Blobs");plt.show()


def km_initializations(random_state = 170):
    # Author: Olivier Grisel <olivier.grisel@ensta.org>
    # License: BSD 3 clause
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_stability_low_dim_dense.html#sphx-glr-auto-examples-cluster-plot-kmeans-stability-low-dim-dense-py
    # Number of run (with randomly generated dataset) for each strategy so as
    # to be able to compute an estimate of the standard deviation
    n_runs = 5

    # k-means models can do several random inits so as to be able to trade
    # CPU time for convergence robustness
    n_init_range = np.array([1, 5, 10, 15, 20])

    # Datasets generation parameters
    n_samples_per_center = 100
    grid_size = 3
    scale = 0.1
    n_clusters = grid_size ** 2


    def make_data(random_state, n_samples_per_center, grid_size, scale):
        random_state = check_random_state(random_state)
        centers = np.array([[i, j]
                            for i in range(grid_size)
                            for j in range(grid_size)])
        n_clusters_true, n_features = centers.shape

        noise = random_state.normal(
            scale=scale, size=(n_samples_per_center, centers.shape[1]))

        X = np.concatenate([c + noise for c in centers])
        y = np.concatenate([[i] * n_samples_per_center
                            for i in range(n_clusters_true)])
        return shuffle(X, y, random_state=random_state)

    # Part 1: Quantitative evaluation of various init methods

    plt.figure()
    plots = []
    legends = []

    cases = [
        (KMeans, 'k-means++', {}),
        (KMeans, 'random', {}),
        (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
        (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
    ]

    for factory, init, params in cases:
        print("Evaluation of %s with %s init" % (factory.__name__, init))
        inertia = np.empty((len(n_init_range), n_runs))

        for run_id in range(n_runs):
            X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
            for i, n_init in enumerate(n_init_range):
                km = factory(n_clusters=n_clusters, init=init, random_state=run_id,
                             n_init=n_init, **params).fit(X)
                inertia[i, run_id] = km.inertia_
        p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
        plots.append(p[0])
        legends.append("%s with %s init" % (factory.__name__, init))

    plt.xlabel('n_init')
    plt.ylabel('inertia')
    plt.legend(plots, legends)
    plt.title("Mean inertia for various k-means init across %d runs" % n_runs)

    # Part 2: Qualitative visual inspection of the convergence

    X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
    km = MiniBatchKMeans(n_clusters=n_clusters, init='random', n_init=1,
                         random_state=random_state).fit(X)

    plt.figure()
    for k in range(n_clusters):
        my_members = km.labels_ == k
        color = cm.nipy_spectral(float(k) / n_clusters, 1)
        plt.plot(X[my_members, 0], X[my_members, 1], 'o', marker='.', c=color)
        cluster_center = km.cluster_centers_[k]
        plt.plot(cluster_center[0], cluster_center[1], 'o',
                 markerfacecolor=color, markeredgecolor='k', markersize=6)
        plt.title("Example cluster allocation with a single random init\n"
                  "with MiniBatchKMeans")
    plt.show()

def km_vs_mbkm():
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py
    from sklearn.datasets.samples_generator import make_blobs
    np.random.seed(0)

    batch_size = 45
    centers = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

    # #############################################################################
    # Compute clustering with Means

    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0

    # #############################################################################
    # Compute clustering with MiniBatchKMeans

    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

    # #############################################################################
    # Plot result

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers,    mbk_means_cluster_centers)

    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('KMeans');    ax.set_xticks(());    ax.set_yticks(())
    plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (   t_batch, k_means.inertia_))

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = mbk_means_labels == order[k]
        cluster_center = mbk_means_cluster_centers[order[k]]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(());    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %             (t_mini_batch, mbk.inertia_))

    # Initialise the different array to all False
    different = (mbk_means_labels == 4)
    ax = fig.add_subplot(1, 3, 3)

    for k in range(n_clusters):
        different += ((k_means_labels == k) != (mbk_means_labels == order[k]))

    identic = np.logical_not(different)
    ax.plot(X[identic, 0], X[identic, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
    ax.plot(X[different, 0], X[different, 1], 'w',   markerfacecolor='m', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(());    ax.set_yticks(());    plt.show()

def sil_based_optimal_km(X = None, y = None):
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.
    if not X or not y:
        X, y = make_blobs(n_samples=500,
                          n_features=2,
                          centers=4,
                          cluster_std=1,
                          center_box=(-10.0, 10.0),
                          shuffle=True,
                          random_state=1)  # For reproducibility

    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()

def compare_linkages(N = 1500, random_state=170):
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html
    from sklearn import datasets
    plot_num = 1
    noisy_circles = datasets.make_circles(n_samples=N, factor=.5, noise=.05)
    noisy_moons = datasets.make_moons(n_samples=N, noise=.05)
    blobs = datasets.make_blobs(n_samples=N, random_state=8)
    no_structure = np.random.rand(N, 2), None
    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=N, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=N, cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

    plt.figure(figsize=(9 * 1.3 + 2, 14.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    default_base = {'n_neighbors': 10, 'n_clusters': 3}

    datasets = [(noisy_circles, {'n_clusters': 2}),(noisy_moons, {'n_clusters': 2}),(varied, {'n_neighbors': 2}), (aniso, {'n_neighbors': 2}),
        (blobs, {}),(no_structure, {})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        X, y = dataset
        X = StandardScaler().fit_transform(X)# normalize dataset for easier parameter selection

        # ============
        # Create cluster objects
        # ============
        ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward')
        complete = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='complete')
        average = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='average')
        single = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='single')

        clustering_algorithms = (('Single Linkage', single),('Average Linkage', average),('Complete Linkage', complete),('Ward Linkage', ward),)

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +" > 1. Completing it to avoid stopping the tree early.",category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']), int(max(y_pred) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5);  plt.ylim(-2.5, 2.5)
            plt.xticks(());  plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15,horizontalalignment='right')
            plot_num += 1
    plt.show()