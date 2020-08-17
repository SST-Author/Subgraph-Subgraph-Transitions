import argparse
import os
import pickle as pkl
import time
import pandas
from typing import List, Tuple, Any

from link_predictor import MatrixLinkPredictor, RandomLinkPredictor, CommonNeighborsStaticLinkPredictor, TGNLinkPredictor
import networkx as nx

import evaluation
from neural_nets.utils import timer
from neural_nets.autoencoders.linear_gae.gae_fit import fit_model as linear_fit_model
from neural_nets.autoencoders.gravity_gae.gae_fit import fit_model as gravity_fit_model
from sst_svm_modeler import SST_SVMLinkPredictor, SST_SVMTemporalLinkPredictor

from multiprocessing import set_start_method

def parse_args() -> Any:
    model_names = ['SST_SVM', 'GCN_AE', 'GCN_VAE', 'Linear_AE', 'Linear_VAE', 'Gravity_GCN_AE', 'Gravity_GCN_VAE', 'Deep_GCN_AE', 'Deep_GCN_VAE', 'TGN', 'Temporal_Graph_Network', 'Random', 'CommonNeighbors']

    partial_runs = ['full', 'count', 'fit', 'eval']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', help='type of input', nargs=1, default='karate')
    parser.add_argument('-j', '--idx', help='idx', nargs=1, default=0, type=int)
    parser.add_argument('-n', '--num_proc', help='num processors (SST only)', nargs=1, default=60, type=int)
    parser.add_argument('-m', '--model', help='type of model', nargs=1, default='SST_SVM', choices=model_names)
    parser.add_argument('-d', '--directed', help='directed?', action='store_true')
    parser.add_argument('-t', '--temporal', help='is temporal?', action='store_true')
    parser.add_argument('-p', '--partial', help='run a subtask (SST only)', nargs=1, default='full', choices=partial_runs)

    return parser.parse_args()

def get_edges(path, directed, temporal, node_to_idx_map) -> List:
    if temporal:
        return get_temporal_edges(path, directed, node_to_idx_map)
    return get_static_edges(path, directed, node_to_idx_map)

def get_static_edges(path, directed, node_to_idx_map) -> List:
    if directed:
        graph_container = nx.DiGraph
    else:
        graph_container = nx.Graph
    g = nx.read_edgelist(path=path, nodetype=int, create_using=graph_container)
    return [(node_to_idx_map[a], node_to_idx_map[b]) for (a, b) in g.edges()]

# Returns list of 4-tuples (source, target, timestamp, weight)
def get_temporal_edges(path, directed, node_to_idx_map) -> List:
    f = open(path, "r")
    edges = []
    for line in f:
        line = line.strip()
        line = line.split(" ")
        assert len(line) == 4
        source = node_to_idx_map[int(line[0])]
        target = node_to_idx_map[int(line[1])]
        timestamp = int(line[2])
        weight = int(line[3])

        if directed:
            edges.append((source, target, timestamp, weight))
        else:
            edges.append((min(source, target), max(source, target), \
                timestamp, weight))
    f.close()
    return edges


def get_nodes(path) -> List:
    with open(path) as f:
        lines = f.readlines()
    return [int(l.strip()) for l in lines]

def get_node_to_idx_map(nodes):
    # Account for the fact that nodes might not be zero-indexed.
    nodes = list(nodes)
    nodes.sort()
    to_idx_map = {}
    for i in range(0, len(nodes)):
        to_idx_map[nodes[i]] = i
    return to_idx_map

def read_auc_edgelists(name: str, is_temporal: bool, idx: int, is_directed: bool) -> Tuple[List, List, List, List,
                                                                                           List, List]:
    """
    Read the nodes list along with the train, validation (true, false), and test (true, false) edgelists
    :return:
    """
    name = name + '_directed' if is_directed else name + '_undirected'
    kind = 'temporal' if is_temporal else 'static'

    nodes_path = os.path.join('./evaluation_data_splits', kind, name, 'node_list.txt')
    nodes = get_nodes(nodes_path)
    node_to_idx_map = get_node_to_idx_map(nodes)
    nodes = [node_to_idx_map[n] for n in nodes]

    conditional_extension = f'_{idx}.txt'
    if is_temporal:
        conditional_extension = '.txt'

    train_edges_path = os.path.join('./evaluation_data_splits', kind, name, f'train_edges{conditional_extension}')
    train_edges = get_edges(train_edges_path, is_directed, is_temporal, node_to_idx_map)

    val_edges_true_path = os.path.join('./evaluation_data_splits', kind, name, f'validation_true_edges{conditional_extension}')
    val_edges_true = get_edges(val_edges_true_path, is_directed, is_temporal, node_to_idx_map)

    val_edges_false_path = os.path.join('./evaluation_data_splits', kind, name, f'validation_false_edges_{idx}.txt')
    val_edges_false = get_edges(val_edges_false_path, is_directed, is_temporal, node_to_idx_map)

    test_edges_true_path = os.path.join('./evaluation_data_splits', kind, name, f'test_true_edges{conditional_extension}')
    test_edges_true = get_edges(test_edges_true_path, is_directed, is_temporal, node_to_idx_map)

    test_edges_false_path = os.path.join('./evaluation_data_splits', kind, name, f'test_false_edges_{idx}.txt')
    test_edges_false = get_edges(test_edges_false_path, is_directed, is_temporal, node_to_idx_map)

    return nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, test_edges_false

def read_aupr_extras(name: str, k: int, is_temporal: bool, idx: int, is_directed: bool) -> Tuple[List, List, List, List]:
    """
    Read the validation (true, false) and test (true, false) edgelists
    :return:
    """
    name = name + '_directed' if is_directed else name + '_undirected'

    kind = 'temporal' if is_temporal else 'static'

    nodes_path = os.path.join('./evaluation_data_splits', kind, name, 'node_list.txt')
    nodes = get_nodes(nodes_path)
    node_to_idx_map = get_node_to_idx_map(nodes)

    if is_temporal:
        extension = f'_k{k}.txt'
    else:
        extension = f'_k{k}_{idx}.txt'
    val_edges_true_path = os.path.join('./evaluation_data_splits', kind, name, f'validation_true_edges{extension}')
    val_edges_true = get_edges(val_edges_true_path, is_directed, is_temporal, node_to_idx_map)

    val_edges_false_path = os.path.join('./evaluation_data_splits', kind, name, f'validation_false_edges{extension}')
    val_edges_false = get_edges(val_edges_false_path, is_directed, is_temporal, node_to_idx_map)

    test_edges_true_path = os.path.join('./evaluation_data_splits', kind, name, f'test_true_edges{extension}')
    test_edges_true = get_edges(test_edges_true_path, is_directed, is_temporal, node_to_idx_map)

    # The test false edge set is _always_ the same for aupr. This is not true of the validation set.
    test_edges_false_path = os.path.join('./evaluation_data_splits', kind, name, f'test_false_edges_k{k}.txt')
    test_edges_false = get_edges(test_edges_false_path, is_directed, is_temporal, node_to_idx_map)

    return val_edges_true, val_edges_false, test_edges_true, test_edges_false

# `aupr_ks` -- list of ints -- can be empty or contain values for AUPR calculation k limits.
def neural_net_runner(name: str, model: str, is_temporal: bool, is_directed: bool, idx: int, aupr_ks: list):
    """
    Runs the link prediction expts on any dataset
    :param method:
    :return:
    """
    nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, \
        test_edges_false = read_auc_edgelists(name=name, idx=idx, is_directed=is_directed, is_temporal=is_temporal)

    aupr_extras = []
    for k in aupr_ks:
        aupr_extras.append((k, read_aupr_extras(name=name, k=k, is_temporal=is_temporal, idx=idx, is_directed=is_directed)))

    g_train = nx.DiGraph() if is_directed else nx.Graph()
    g_train.add_nodes_from(nodes)  # add the nodes
    g_train.add_edges_from(train_edges)  # add the edges
    adj_train = nx.adjacency_matrix(g_train)

    if 'gravity' in model or 'Gravity' in model:
        output_mat = gravity_fit_model(adj=adj_train, val_edges=val_edges_true,
                        val_edges_false=val_edges_false, test_edges=test_edges_true,
                        test_edges_false=test_edges_false, model_name=model)
    else:
        output_mat = linear_fit_model(adj=adj_train, val_edges=val_edges_true,
                        val_edges_false=val_edges_false, test_edges=test_edges_true,
                        test_edges_false=test_edges_false, model_name=model)

    matrix_link_predictor = MatrixLinkPredictor(output_mat, nodes)

    evaluation.score_link_predictor(matrix_link_predictor, test_edges_true, test_edges_false, aupr_extras, \
        name, is_directed, is_temporal, idx, model)

    return

# `aupr_ks` -- list of ints -- can be empty or contain values for AUPR calculation k limits.
def other_runner(name: str, model: str, is_temporal: bool, is_directed: bool, idx: int, aupr_ks: list, partial: str, num_proc: int):

    nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, \
        test_edges_false = read_auc_edgelists(name=name, idx=idx, is_directed=is_directed, is_temporal=is_temporal)
    aupr_extras = []
    for k in aupr_ks:
        aupr_extras.append((k, read_aupr_extras(name=name, k=k, is_temporal=is_temporal, idx=idx, is_directed=is_directed)))

    if model == 'Random':
        predictor = RandomLinkPredictor(nodes, train_edges)

        evaluation.score_link_predictor(predictor, test_edges_true, test_edges_false, aupr_extras, \
            name, is_directed, is_temporal, idx, model)
    elif model == 'CommonNeighbors':
        assert not is_temporal
        predictor = CommonNeighborsStaticLinkPredictor(nodes, train_edges, directed=is_directed)

        evaluation.score_link_predictor(predictor, test_edges_true, test_edges_false, aupr_extras, \
            name, is_directed, is_temporal, idx, model)
    else:
        subgraph_size = 4
        assert model == "SST_SVM"

        if subgraph_size != 4:
            model = model + "_%d" % subgraph_size

        count = partial == 'full' or partial == 'count'
        fit = partial == 'full' or partial == 'fit'
        evaluate = partial == 'full' or partial == 'eval'

        if count:
            if is_temporal:
                if is_directed:
                    non_edge_mult = 1
                else:
                    non_edge_mult = 1
                predictor = SST_SVMTemporalLinkPredictor(nodes, train_edges + val_edges_true, directed=is_directed, num_processes=num_proc, \
                    non_edge_multiplier=non_edge_mult, subgraph_size=subgraph_size, scale_data=False)
                predictor.become_serializeable()
                evaluation.save_model_data(predictor, "with_counts", model, name, is_temporal, is_directed, idx)
            else:
                predictor = SST_SVMLinkPredictor(nodes, train_edges + val_edges_true, directed=is_directed, num_processes=num_proc, \
                    non_edge_multiplier=10, subgraph_size=subgraph_size, scale_data=False)
                predictor.become_serializeable()
                evaluation.save_model_data(predictor, "with_counts", model, name, is_temporal, is_directed, idx)

        if fit and not count:
            predictor = evaluation.load_model_data("with_counts", model, name, is_temporal, is_directed, idx)
        if fit:
            predictor.fit()
            predictor.become_serializeable()
            evaluation.save_model_data(predictor, "fitted", model, name, is_temporal, is_directed, idx)
            ssts = predictor.get_interpretable_model()
            evaluation.save_result("ssts", ssts, name, is_temporal, is_directed, idx, model)

        if evaluate and not fit:
            predictor = evaluation.load_model_data("fitted", model, name, is_temporal, is_directed, idx)

        if evaluate:
            evaluation.score_link_predictor(predictor, test_edges_true, test_edges_false, aupr_extras, \
                name, is_directed, is_temporal, idx, model)

    return

def temporal_net_runner(name: str, model: str, is_temporal: bool, is_directed: bool, idx: int, aupr_ks: list):
    """
    Runs the link prediction expts on any dataset
    :param method:
    :return:
    """
    from neural_nets.tgn.tgn_fit import fit_model as tgn_fit_model

    # read in the data split
    nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, test_edges_false \
            = read_auc_edgelists(name=name, idx=idx, is_directed=is_directed, is_temporal=is_temporal)

    # reindex data for PyTorch
    nodes = [n + 1 for n in nodes]
    train_edges = [(u+1, v+1, ts+1, idx) for idx, (u, v, ts, w) in enumerate(train_edges, 1)]
    max_idx = len(train_edges)
    val_edges_true = [(u+1, v+1, ts+1, idx) for idx, (u, v, ts, w) in enumerate(val_edges_true, max_idx + 1)]
    max_idx += len(val_edges_true)
    test_edges_true = [(u+1, v+1, ts+1, idx) for idx, (u, v, ts, w) in enumerate(test_edges_true, max_idx + 1)]
    max_idx += len(test_edges_true)
    val_edges_false = [(u+1, v+1, ts+1, idx) for idx, (u, v, ts, w) in enumerate(val_edges_false, max_idx + 1)]
    max_idx += len(val_edges_false)
    test_edges_false = [(u+1, v+1, ts+1, idx) for idx, (u, v, ts, w) in enumerate(test_edges_false, max_idx + 1)]
    max_idx += len(test_edges_false)

    # create the full dataframe
    full_edges = train_edges + val_edges_true + test_edges_true
    rows = {'u': [], 'i': [], 'ts': [], 'label': [], 'idx': []}
    for u, i, ts, idx in full_edges:
        rows['u'].append(u)
        rows['i'].append(i)
        rows['ts'].append(ts)
        rows['label'].append(0.0)
        rows['idx'].append(idx)
    full_df = pandas.DataFrame(rows)

    aupr_extras = []
    for k in aupr_ks:
        foo = read_aupr_extras(name=name, k=k, is_temporal=is_temporal, idx=idx, is_directed=is_directed)
        bar = []
        for chunk in foo:
            temp = [(u+1, v+1, ts+1, idx) for idx, (u, v, ts, _) in enumerate(chunk, max_idx + 1)]
            max_idx += len(temp)
            bar.append(temp)
        aupr_extras.append((k, tuple(bar)))

    # train the TGN model on the data split
    tgn, neg_sampler = tgn_fit_model(full_df, nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, test_edges_false, max_idx)

    tgn_link_predictor = TGNLinkPredictor(tgn=tgn, neg_sampler=neg_sampler)

    evaluation.score_link_predictor(tgn_link_predictor, test_edges_true, test_edges_false, aupr_extras, name, is_directed, is_temporal, idx, model)

    return

if __name__ == '__main__':

    # Allows for better use of python's pools.
    # set_start_method("spawn")

    args = parse_args()
    inp, idx, model, is_directed, is_temporal, partial, num_proc = \
        args.input, args.idx, args.model, args.directed, args.temporal, args.partial, args.num_proc

    if type(inp) is list:
        inp = inp[0]
    if type(idx) is list:
        idx = idx[0]
    if type(model) is list:
        model = model[0]
    if type(is_directed) is list:
        is_directed = is_directed[0]
    if type(is_temporal) is list:
        is_temporal = is_temporal[0]
    if type(partial) is list:
        partial = partial[0]
    if type(num_proc) is list:
        num_proc = num_proc[0]

    # aupr_ks = [2, 3]
    aupr_ks = [3]
    if inp == "wiki-en-additions" or inp.startswith("BA_"):
        aupr_ks = []

    if model in ['SST_SVM', 'Random', 'CommonNeighbors']:
        other_runner(name=inp, model=model, is_temporal=is_temporal, is_directed=is_directed, idx=idx, aupr_ks=aupr_ks, partial=partial, num_proc=num_proc)
    elif model in ['GCN_AE', 'GCN_VAE', 'Linear_AE', 'Linear_VAE', 'Deep_GCN_AE', 'Deep_GCN_VAE', 'Gravity_GCN_AE', 'Gravity_GCN_VAE']:
        neural_net_runner(name=inp, model=model, is_temporal=is_temporal, is_directed=is_directed, idx=idx, aupr_ks=aupr_ks)
    elif model in ['Temporal_Graph_Network', 'TGN']:
        temporal_net_runner(name=inp, model=model, is_temporal=True, is_directed=True, idx=idx, aupr_ks=aupr_ks)
