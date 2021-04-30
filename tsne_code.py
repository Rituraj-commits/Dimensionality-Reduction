import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tsnecuda import TSNE
import time
import seaborn as sns
import matplotlib.patheffects as PathEffects
'exec(%matplotlib inline)'
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})




def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

x_train, y_train = load_mnist('Downloads/fashion_mnist', kind='train')

print(x_train.shape)
print(y_train.shape)


def tsne_plot(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    
    plt.show()

    return f, ax, sc, txts


X = x_train[0:20000]
Y = y_train[0:20000]

time_start = time.time()

from sklearn.decomposition import PCA
pca = PCA(n_components = 2) 
X = pca.fit_transform(X)

print(pca.explained_variance_ratio_)

print("Time elapsed: ",time.time()-time_start)

tsne_plot(X,Y)

time_start = time.time()

tsne = TSNE().fit_transform(X)


print("Time elapsed: ",time.time()-time_start)

