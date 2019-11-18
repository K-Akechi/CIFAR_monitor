import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time


n = 10

interValues_train = np.load('training_set_neuron_outputs.npy')
labels_train = np.load('training_set_labels.npy')
interValues_test = np.load('test_set_neuron_outputs.npy')
labels_test = np.load('test_set_labels.npy')
predictions_test = np.load('test_set_predictions.npy')
print('data retrieve success.')

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']
start_time = time.time()
pca = TSNE(n_components=2)
pca_reduction = pca.fit_transform(interValues_train)
fig = plt.figure(1)
# ax = Axes3D(fig)
for i in range(10):
    print(pca_reduction[labels_train == i, 0].shape)
    plt.scatter(pca_reduction[labels_train == i, 0], pca_reduction[labels_train == i, 1],
                c=color[i], marker='o', s=2, linewidths=0, alpha=0.8)

# tsne = TSNE(n_components=2)
# tsne_reduction = tsne.fit_transform(interValues_train)
# plt.figure(2)
# for i in range(10):
#     print(tsne_reduction[labels_train == i, 0].shape)
#     plt.scatter(tsne_reduction[labels_train == i, 0], tsne_reduction[labels_train == i, 1], c=color[i], marker='o',
#                 s=2, linewidths=0, alpha=0.8, label='%s' % i)

print('{}seconds.'.format(time.time()-start_time))
# plt.savefig('test2.png', bbox_inches='tight')
plt.show()
