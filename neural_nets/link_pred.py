import logging
from typing import Tuple, List

import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

from methods.VRG.runner import get_grammars, get_graph, make_dirs
from methods.VRG.src.VRG import NCE
from methods.VRG.src.generate import GreedyGenerator, NCEGenerator, EnsureAllNodesGenerator
from methods.VRG.src.utils import check_file_exists, load_pickle, dump_pickle
from methods.autoencoders.linear_gae.gae_fit import fit_model
from utils import sparse_to_tuple, make_plot, sigmoid


class LinkPrediction:
    """

    """
    METRICS = 'AUPR', 'AUROC', 'ACC', 'AP', 'F1'

    def __init__(self, input_graph: nx.Graph, test_valid_split: Tuple[float, float],
                 dataset: str, use_pickle: bool = False):
        self.input_graph = input_graph
        self.dataset = dataset
        self.method = None
        self.test_frac, self.valid_frac = test_valid_split

        self.adj_train, self.train_edges, self.train_edges_false, self.val_edges, self.val_edges_false, \
            self.test_edges, self.test_edges_false = self._partition_graph(test_frac=self.test_frac,
                                                                           val_frac=self.valid_frac,
                                                                           verbose=True, use_pickle=use_pickle)

        self.performance = {metric: np.nan for metric in LinkPrediction.METRICS}
        return

    def set_method(self, method):
        self.method = method
        return

    def _partition_graph(self, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False, use_pickle=False):
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        # taken from https://github.com/lucashu1/link-prediction/blob/master/gae/preprocessing.py

        splits_filename = f'./dumps/splits/{self.dataset}_{int(test_frac*100)}_{int(val_frac*100)}.pkl'
        if use_pickle and check_file_exists(splits_filename):
            logging.error(f'Using pickle at {splits_filename!r}')
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_pickle(splits_filename)
        else:
            g = nx.Graph(self.input_graph)
            adj = nx.to_scipy_sparse_matrix(g)
            orig_num_cc = nx.number_connected_components(g)

            adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
            adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
            edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
            # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
            num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
            num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

            # Store edges in list of ordered tuples (node1, node2) where node1 < node2
            edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
            all_edge_tuples = set(edge_tuples)
            train_edges = set(edge_tuples)  # initialize train_edges to have all edges
            test_edges = set()
            val_edges = set()

            if verbose:
                print('generating test/val sets...', end=' ')

            # Iterate over shuffled edges, add to train/val sets
            np.random.shuffle(edge_tuples)
            for edge in edge_tuples:
                # print edge
                node1 = edge[0]
                node2 = edge[1]

                # If removing edge would disconnect a connected component, backtrack and move on
                g.remove_edge(node1, node2)
                if prevent_disconnect:
                    if nx.number_connected_components(g) > orig_num_cc:
                        g.add_edge(node1, node2)
                        continue

                # Fill test_edges first
                if len(test_edges) < num_test:
                    test_edges.add(edge)
                    train_edges.remove(edge)

                # Then, fill val_edges
                elif len(val_edges) < num_val:
                    val_edges.add(edge)
                    train_edges.remove(edge)

                # Both edge lists full --> break loop
                elif len(test_edges) == num_test and len(val_edges) == num_val:
                    break

            if (len(val_edges) < num_val) or (len(test_edges) < num_test):
                print("WARNING: not enough removable edges to perform full train-test split!")
                print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
                print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

            if prevent_disconnect:
                assert nx.number_connected_components(g) == orig_num_cc

            if verbose:
                print('creating false test edges...', end=' ')

            test_edges_false = set()
            while len(test_edges_false) < num_test:
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue

                false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

                # Make sure false_edge not an actual edge, and not a repeat
                if false_edge in all_edge_tuples:
                    continue
                if false_edge in test_edges_false:
                    continue

                test_edges_false.add(false_edge)

            if verbose:
                print('creating false val edges...', end=' ')

            val_edges_false = set()
            while len(val_edges_false) < num_val:
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue

                false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

                # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
                if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false:
                    continue

                val_edges_false.add(false_edge)

            if verbose:
                print('creating false train edges...')

            train_edges_false = set()
            while len(train_edges_false) < len(train_edges):
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue

                false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

                # Make sure false_edge in not an actual edge, not in test_edges_false,
                # not in val_edges_false, not a repeat
                if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false or \
                        false_edge in train_edges_false:
                    continue

                train_edges_false.add(false_edge)

            if verbose:
                print('final checks for disjointness...', end=' ')

            # assert: false_edges are actually false (not in all_edge_tuples)
            assert test_edges_false.isdisjoint(all_edge_tuples)
            assert val_edges_false.isdisjoint(all_edge_tuples)
            assert train_edges_false.isdisjoint(all_edge_tuples)

            # assert: test, val, train false edges disjoint
            assert test_edges_false.isdisjoint(val_edges_false)
            assert test_edges_false.isdisjoint(train_edges_false)
            assert val_edges_false.isdisjoint(train_edges_false)

            # assert: test, val, train positive edges disjoint
            assert val_edges.isdisjoint(train_edges)
            assert test_edges.isdisjoint(train_edges)
            assert val_edges.isdisjoint(test_edges)

            if verbose:
                print('creating adj_train...', end=' ')

            # Re-build adj matrix using remaining graph
            adj_train = nx.adjacency_matrix(g)

            # Convert edge-lists to numpy arrays
            train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
            train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
            val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
            val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
            test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
            test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

            if verbose:
                print('Done with train-test split!', end=' ')
                print()

            # NOTE: these edge lists only contain single direction of edge!
            dump_pickle((adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), splits_filename)
        logging.error(f'train (T/F): {len(train_edges)} valid: {len(val_edges)} ({val_frac*100}%) test: {len(test_edges)} ({test_frac*100}%)')
        return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

    def evaluate(self, predicted_adj_mat: np.array, make_plots: bool = True):
        """
        Evaluate the performance of the link prediction algorithm with the generated graph
        fill the
        :return:
        """
        true_edges, false_edges = [], []
        y_score = []

        for i, (u, v) in enumerate(self.test_edges):
            score = predicted_adj_mat[u, v]
            y_score.append(score)  # if there's an edge - it'll be 1 or close to 1
            true_edges.append(1)  # actual edge

        for i, (u, v) in enumerate(self.test_edges_false):
            score = predicted_adj_mat[u, v]
            y_score.append(score)  # the numbers should be 0 or close to 0
            false_edges.append(0)  # actual non-edge

        y_true = true_edges + false_edges
        assert len(y_score) == len(y_true), f'Lengths of y_score: {len(y_score)!r} y_true: {len(y_true)!r} not equal'

        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
        auroc = roc_auc_score(y_true=y_true, y_score=y_score)

        prec, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
        aupr = auc(recall, prec, reorder=True)

        self.performance['AUROC'] = auroc
        self.performance['AUPR'] = aupr
        self.performance['AP'] = average_precision_score(y_true=y_true, y_score=y_score)
        # self.performance['ACC'] = accuracy_score(y_true=y_true, y_pred=y_score)
        # self.performance['F1'] = f1_score(y_true=y_true, y_pred=y_score)

        if make_plots:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            make_plot(x=fpr, y=tpr, xlabel='False Positive Rate', ylabel='True Positive Rate',
                      c='darkorange', label=f'AUROC {round(auroc, 2)}',
                      title=f'{self.method!r} {self.dataset!r} ROC curve', ax=ax1)

            make_plot(x=recall, y=prec, xlabel='Recall', ylabel='Precision', c='darkorange',
                      label=f'AUPR {round(aupr, 2)}',
                      title=f'{self.method!r} {self.dataset!r} PR curve', ax=ax2, kind='pr')
            plt.show()
        return

    def __str__(self):
        st = f'\n<dataset: {self.dataset!r} method: {self.method!r} test: {self.test_frac*100}% valid: {self.valid_frac*100}%'
        perf = {k: np.round(v, 3) for k, v in self.performance.items() if not np.isnan(v)}
        st += f' performance: {perf!r}>'
        return st


def combine_graphs_into_matrices(graphs: List[nx.Graph]) -> np.array:
    combined_adj_mat = None
    nodelist = sorted(graphs[0].nodes)
    for g in graphs:
        adj_mat = nx.adjacency_matrix(g, nodelist=nodelist)
        if combined_adj_mat is None:
            combined_adj_mat = adj_mat
        else:
            combined_adj_mat += adj_mat

    combined_adj_mat = combined_adj_mat / len(graphs)
    return combined_adj_mat


def vrg_runner(link_pred_obj: LinkPrediction, count: int, mu: int) -> np.array:
    """
    Runs VRG on the input graph and returns the combined adjacency matrix
    """
    link_pred.set_method(method=f'VRG_{mu}')
    train_g: nx.Graph = nx.from_scipy_sparse_matrix(link_pred_obj.adj_train, create_using=nx.Graph)
    train_g.add_edges_from(link_pred_obj.val_edges.tolist())   # add the validation edges too

    vrg: NCE = get_grammars(clustering='leiden', grammar_type='NCE', mu=mu, name=name,
                            input_graph=train_g, use_pickle=False)[0]
    gen = EnsureAllNodesGenerator(grammar=vrg, input_graph=train_g, fraction=1)
    gen_graphs = gen.generate(count)
    combined_adj_matrices = combine_graphs_into_matrices(gen_graphs)

    return combined_adj_matrices


def autoencoder_runner(model: str, link_pred_obj: LinkPrediction) -> np.array:
    link_pred.set_method(method=model)
    mat = fit_model(adj=link_pred_obj.adj_train, val_edges=link_pred_obj.val_edges,
                    val_edges_false=link_pred_obj.val_edges_false, test_edges=link_pred_obj.test_edges,
                    test_edges_false=link_pred_obj.test_edges_false, model_name=model)
    return mat


def basic_runner(link_pred_obj: LinkPrediction, kind: str) -> np.array:
    link_pred.set_method('Jaccard')
    train_g: nx.Graph = nx.from_scipy_sparse_matrix(link_pred_obj.adj_train, create_using=nx.Graph)
    train_g.add_edges_from(link_pred_obj.val_edges.tolist())  # add the validation edges too

    pred_mat = np.zeros((train_g.order(), train_g.order()))
    only_compute = link_pred_obj.test_edges.tolist() + link_pred_obj.test_edges_false.tolist()
    if kind == 'jaccard':
        func = nx.jaccard_coefficient
    elif kind == 'adamic-adar':
        func = nx.adamic_adar_index
    else:
        raise NotImplementedError()
    for u, v, d in func(train_g, only_compute):
        pred_mat[u, v] = d
    return pred_mat


# TODO: clean this up
# TODO: get this to work with train/val/test splits
if __name__ == '__main__':
    #name = 'gnutella'
    name = 'cora.cites'
    #orig_g = get_graph(name)
    orig_g = nx.read_edgelist(f'autoencoders/data/{name}', create_using=nx.DiGraph())
    #exit(1)
    #model = 'jaccard'
    model = 'gravity_ae'
    #trials = 10
    trails = 1
    print(name, model)
    for trial in range(1, trials+1):
        link_pred = LinkPrediction(input_graph=orig_g, test_valid_split=(0.1, 0.05),
                                   dataset=name, use_pickle=True)
        if 'ae' in model:
            pred_mat = autoencoder_runner(model=model, link_pred_obj=link_pred)
            dump_pickle(pred_mat, f'./dumps/predictions/autoencoders/{name}_{trial}_{model}.pkl')
        elif 'vrg' in model:
            pred_mat = vrg_runner(link_pred_obj=link_pred, count=20, mu=5)
            dump_pickle(pred_mat, f'./dumps/predictions/vrg/{name}_{trial}.pkl')
        else:
            pred_mat = basic_runner(link_pred_obj=link_pred, kind=model)
            dump_pickle(pred_mat, f'./dumps/predictions/basic/{name}_{model}.pkl')
            break  # jaccard only once
    # link_pred.evaluate(predicted_adj_mat=pred_mat)
    # print(link_pred)

    # link_pred = LinkPrediction(input_graph=orig_g, test_valid_split=(0.1, 0.05),
    #                            dataset=name, use_pickle=True)
    # vrg_mat = vrg_runner(link_pred_obj=link_pred, count=20)
    # link_pred.evaluate(predicted_adj_mat=vrg_mat)
    #
    # print('VRG', link_pred)
