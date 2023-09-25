import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_s_curve
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from numpy import loadtxt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.preprocessing import StandardScaler

n_samples = 1500
S_points, S_color = make_s_curve(n_samples, random_state=0)

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(30, 30),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=10, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
    plt.savefig('results/tripple_lost_class_iden/plot_3d.png')


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(30, 30), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()
    plt.savefig('results/tripple_lost_class_iden/plot_2d.png')


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)

def KPCA(X, d, kernel):
  [n,D] = X.shape
  K = pdist(X, lambda x, y: kernel(x,y))
  K = squareform(K)
  J = np.eye(n) - (1/n)*np.ones((n,n))
  center_k = J@K@J
  eigvals, eigvecs = np.linalg.eigh(center_k)
  eigvecs = eigvecs[:,::-1][:,0:d]
  eigvals = eigvals[::-1][0:d]
  gama_inv = np.linalg.inv(np.diag((eigvals)**(0.5)))
  return (gama_inv@eigvecs.T@center_k).T

# plot_3d(S_points, S_color, "Original S-curve samples")

def tsne(data_net_outputs_path_csv,
         labels_net_outputs_path_csv,
         toScale):
    data = loadtxt(data_net_outputs_path_csv)
    labels = loadtxt(labels_net_outputs_path_csv)
    if toScale==True:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    t_sne2 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10)
    tsne2d = t_sne2.fit_transform(np.asarray(data))
    plot_2d(tsne2d, labels, "TSNE 2D")
    t_sne3 = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=36)
    tsne3d = t_sne3.fit_transform(np.asarray(data))
    plot_3d(tsne3d, labels, "TSNE 3D")


def main():
    tsne('./results/NET_OUTPUT_resnet_triple.csv',
         './results/labels_resnet_triple.csv',
         toScale=True)

if __name__ == '__main__':
    main()
