import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics, mixture, cluster
import time


def kmeans(samples, n_clusters, samples_to_predict):
    km = cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)
    km.fit(samples)
    return km.predict(samples), km.predict(samples_to_predict)


def gaussian(samples, n_clusters, samples_to_predict):
    gmm = mixture.GaussianMixture(n_components=n_clusters)
    gmm.fit(samples)
    return gmm.predict(samples), gmm.predict(samples_to_predict)


def affinitypropagation(samples, samples_to_predict):
    aff = cluster.AffinityPropagation()
    aff.fit(samples)
    return aff.predict(samples), aff.predict(samples_to_predict)


def birch(samples, n_clusters, samples_to_predict):
    brc = cluster.Birch(n_clusters=n_clusters)
    brc.fit(samples)
    return brc.predict(samples), brc.predict(samples_to_predict)


if __name__ == '__main__':
    n = 10
    vgg_path = './vgg_intervalues/'
    interValues_train = np.load(vgg_path+'training_set_neuron_outputs.npy')
    labels_train = np.load(vgg_path+'training_set_labels.npy')
    interValues_test = np.load(vgg_path+'test_set_neuron_outputs.npy')
    labels_test = np.load(vgg_path+'test_set_labels.npy')
    predictions_test = np.load(vgg_path+'test_set_predictions.npy')
    print('data retrieve success.')

    stat = np.zeros((n, 10), dtype=int)
    stat_test = np.zeros((n, 10), dtype=int)
    start_time = time.time()

    kmeans_train_result, kmeans_test_result = kmeans(interValues_train, n, interValues_test)
    # gmm_train_result, gmm_test_result = gaussian(interValues_train, n, interValues_test)
    # birch_train_result, birch_test_result = birch(interValues_train, n, interValues_test)
    # aff_train_result, aff_test_result = affinitypropagation(interValues_train, interValues_test)
    duration = time.time() - start_time
    print('clustering finish in {} seconds'.format(duration))

    for i in range(interValues_train.shape[0]):
        stat[kmeans_train_result[i]][labels_train[i]] += 1
    print(stat)
    index = np.argmax(stat, axis=1)
    print(index)

    correct = 0
    out_of_cluster = 0
    ooc_and_misclassified = 0
    for i in range(interValues_test.shape[0]):
        if predictions_test[i] == labels_test[i]:
            correct += 1
        if index[kmeans_test_result[i]] != predictions_test[i]:
            out_of_cluster += 1
            if predictions_test[i] != labels_test[i]:
                ooc_and_misclassified += 1

    print(ooc_and_misclassified, out_of_cluster, interValues_test.shape[0] - correct, correct)
