import functools
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        ColorPrint.print_bold(f'Start: {datetime.now().ctime()}')
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        ColorPrint.print_bold(f'End: {datetime.now().ctime()}')
        ColorPrint.print_bold(f'Elapsed time: {elapsed_time:0.4f} seconds')
        return value
    return wrapper_timer

# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def make_plot(x, y, ax, xlabel=None, ylabel=None, c=None, label=None, title=None, kind=None):
    ax.plot(x, y, marker='o', linestyle='--', c=c, label=label)
    if kind is None:
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_prob_mat_from_emb(emb):
    """
    Get adjacency matrix from embeddings - hard threshold
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :return: return adj matrix
    """
    mat = sigmoid(emb.dot(emb.T))
    np.fill_diagonal(mat, 0)
    mat = np.maximum(mat, mat.T)  # make it symmetric
    return mat

