from link_sample_utils import *
import networkx as nx
import numpy as np
import os
import sys

# Usage: python evaluation_data_splits.py <filepath> <temporal> <directed> <k for AUPR> <num_splits> <overwrite>
#
# filepath -- path to the .g file to be turned into an evaluation dataset
#
# temporal -- if the graph is to be treated as a temporal network
#   values:
#       temporal
#       static
#
# directed -- if the graph is to be treated as a directed or undirected network
#   values:
#       directed
#       undirected
#
# k for AUPR -- if area under precision recall curve is to be calculated, we
#       must not downsample negative edges. This leads to a large evaluation
#       burden, so we restrict the task to testing on all non-edges where the
#       endpoints are within k hops of each other. If no AUPR is to be
#       calculated, just pass no_aupr.
#   values:
#       no_aupr
#       aupr_k=<some integer > 1>
#       aupr_k=<some integer > 1>:<some integer greater than before>
#
# num_splits -- number of splits into different tests
#   values:
#       <some integer >= 1>
#
# overwrite -- if set, replaces existing files with same name as output
#   values:
#       overwrite
#       no_overwrite

# Returns true if everything prepared - false otherwise - thus can return
#   false if file exists and overwrite=False.
def __prepare_path__(filepath, overwrite=True):
    if os.path.isfile(filepath):
        return overwrite
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return True

def __export_items_list__(filepath, items_list, overwrite=True):
    if not __prepare_path__(filepath, overwrite=overwrite):
        print("Skipping writing file %s. Already exists and overwrite=False."%\
            filepath)
        return

    f = open(filepath, "w")
    for item in items_list:
        if type(item) is int:
            f.write(str(item) + "\n")
        elif type(item) is tuple:
            for i in range(0, len(item)):
                f.write(str(item[i]))
                if i < len(item) - 1:
                    f.write(" ")
            f.write("\n")
    f.close()


# Reads the edge list with potentially duplicated edges and returns two lists:
#   nodes and edges
def __read_edge_list__(filename, directed, temporal):
    f = open(filename, "r")
    edges = set()
    nodes = set()
    self_loops = 0
    for line in f:
        line = line.strip()
        line = line.split(" ")
        assert len(line) == 2 + int(temporal)
        source = int(line[0])
        target = int(line[1])

        nodes.add(source)
        nodes.add(target)

        if source == target:
            self_loops += 1
            continue

        if directed:
            edge = [source, target]
        else:
            edge = [min(source, target), max(source, target)]

        if temporal:
            timestamp = line[2]
            if timestamp.isnumeric():
                timestamp = int(timestamp)
            edge.append(timestamp)
        edges.add(tuple(edge))

    f.close()
    print("Input file had %d self-loops, all of which (if any) were removed." %\
        self_loops)
    return (list(nodes), list(edges))

if __name__ == "__main__":

    assert len(sys.argv) == 7

    filepath = sys.argv[1]
    temporal = sys.argv[2]
    directed = sys.argv[3]
    k_for_aupr = sys.argv[4]
    num_splits = sys.argv[5]
    overwrite = sys.argv[6]

    assert os.path.isfile(filepath)
    assert filepath.endswith(".g")
    assert temporal in ["temporal", "static"]
    assert directed in ["directed", "undirected"]
    assert k_for_aupr == "no_aupr" or \
        (k_for_aupr.startswith("aupr_k=") and len(k_for_aupr) > 7)
    assert overwrite in ["overwrite", "no_overwrite"]

    temporal = temporal == "temporal"
    directed = directed == "directed"
    num_splits = int(num_splits)
    overwrite = overwrite == "overwrite"

    # Changed later unless k_for_aupr == "no_aupr"
    aupr = k_for_aupr != "no_aupr"

    if aupr:
        k_for_aupr = str(k_for_aupr[7:])
        k_for_aupr = k_for_aupr.split(":")
        assert len(k_for_aupr) >= 1 and len(k_for_aupr) <= 2
        k_for_aupr = [int(s) for s in k_for_aupr]
        if len(k_for_aupr) == 1:
            k_for_aupr_range = [k_for_aupr[0], k_for_aupr[0] + 1]
        else:
            assert k_for_aupr[0] < k_for_aupr[1]
            k_for_aupr_range = k_for_aupr

    (_, graph_name) = os.path.split(filepath)

    # Valid due to assertion that extension is ".g"
    graph_name = str(graph_name[:-2])

    # Load graph.
    (nodes, edges) = __read_edge_list__(filepath, directed, temporal)
    if not temporal:
        if directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        for node in nodes:
            graph.add_node(node)
        for (a, b) in edges:
            graph.add_edge(a, b)

    if temporal:
        edges = [(t, a, b) for (a, b, t) in edges]
        edges.sort()
        edges = [(a, b, t) for (t, a, b) in edges]

    if temporal:
        # TODO: Consider forcing chunk splits to be at timestamp changes.
        num_chunks = 10

        chunk_size = int(len(edges) / num_chunks)
        split_indices = [0] + [len(edges) - (chunk_size * (num_chunks - i)) \
            for i in range(1, num_chunks + 1)]
        test_size = chunk_size
        valid_size = chunk_size
        train_size = len(edges) - 2 * chunk_size

        re_timestamped_chunks = [[(a, b, i) for (a, b, t) in \
            edges[split_indices[i]:split_indices[i+1]]] \
                for i in range(0, num_chunks)]

        # Remove duplicates inside a chunk and add weights based on the number
        #   of occurrences per chunk.
        for chunk_idx in range(0, len(re_timestamped_chunks)):
            edge_weights = {}
            # Here t is same as chunk idx.
            for edge in re_timestamped_chunks[chunk_idx]:
                assert len(edge) == 3
                if edge not in edge_weights:
                    edge_weights[edge] = 1
                else:
                    edge_weights[edge] += 1
            re_timestamped_chunks[chunk_idx] = \
                [(a, b, t, w) for ((a, b, t), w) in edge_weights.items()]

        temporal_train_edges = re_timestamped_chunks[0]
        for i in range(1, num_chunks - 2):
            temporal_train_edges += re_timestamped_chunks[i]
        temporal_valid_edges = re_timestamped_chunks[-2]
        temporal_test_edges = re_timestamped_chunks[-1]

        print("Before collapsing repeats: train, validate, test: %d, %d, %d" % \
            (train_size, valid_size, test_size))
        train_size = len(temporal_train_edges)
        valid_size = len(temporal_valid_edges)
        test_size = len(temporal_test_edges)
        print("After collapsing repeats: train, validate, test: %d, %d, %d" % \
            (train_size, valid_size, test_size))

        temporal_valid_edges_no_tw=[(a,b) for (a,b,t,w) in temporal_valid_edges]
        temporal_test_edges_no_tw = [(a,b) for (a,b,t,w) in temporal_test_edges]

    else:  # static
        # 85, 5, 10 split.
        test_size = int(len(edges) * 0.1)
        valid_size = int(len(edges) * 0.05)
        train_size = len(edges) - (test_size + valid_size)
        print("len(edges): %d, train: %d, valid: %d, test: %d" % (len(edges), train_size, valid_size, test_size))

    if aupr and temporal:
        # Compute negative edge sets. They are fixed. Only true sets change.

        # Here, limiting by 'predict for disconnected nodes within k hops'
        #   makes less sense given that previous edges are not permanent.
        #   however, I will treat them as permanent. The valid hypothesis space
        #   then becomes:
        #       {all previous interactions where, if disconnected*, would be
        #           within k hops*} union
        #       {all non-previous interactions where nodes are within k hops*}
        #
        #   *in the graph of all previous interactions.

        if directed:
            g_prime = nx.DiGraph()
        else:
            g_prime = nx.Graph()
        for node in nodes:
            g_prime.add_node(node)
        for (a, b, t, w) in temporal_train_edges:
            g_prime.add_edge(a, b)

        aupr_valid_false_edges = {}
        aupr_valid_true_edges = {}
        for k in range(k_for_aupr_range[0], k_for_aupr_range[1]):
            hypothesis_space = []
            hypothesis_space += \
all_connected_node_pairs_that_would_be_within_k_if_disconnected(g_prime, k)
            hypothesis_space += \
                all_disconnected_node_pairs_within_k(g_prime, k)

            if not directed:
                hypothesis_space = \
                    set([(min(a,b), max(a,b)) for (a, b) in hypothesis_space])
            else:
                hypothesis_space = set(hypothesis_space)

            aupr_valid_true_edges[k] = []
            for (a, b, t, w) in temporal_valid_edges:
                if (a, b) in hypothesis_space:
                    aupr_valid_true_edges[k].append((a, b, t, w))

            (_, _, t, _) = temporal_valid_edges[0]
            ab = hypothesis_space - set(temporal_valid_edges_no_tw)
            aupr_valid_false_edges[k] = [(a, b, t, 1) for (a, b) in ab]

        # Add the valid edges and do the same thing to create test sets.
        for (a, b, t, w) in temporal_valid_edges:
            g_prime.add_edge(a, b)

        aupr_test_false_edges = {}
        aupr_test_true_edges = {}
        for k in range(k_for_aupr_range[0], k_for_aupr_range[1]):
            hypothesis_space = []
            hypothesis_space += \
all_connected_node_pairs_that_would_be_within_k_if_disconnected(g_prime, k)
            hypothesis_space += \
                all_disconnected_node_pairs_within_k(g_prime, k)

            if not directed:
                hypothesis_space = \
                    set([(min(a,b), max(a,b)) for (a, b) in hypothesis_space])
            else:
                hypothesis_space = set(hypothesis_space)

            aupr_test_true_edges[k] = []
            for (a, b, t, w) in temporal_test_edges:
                if (a, b) in hypothesis_space:
                    aupr_test_true_edges[k].append((a, b, t, w))

            (_, _, t, _) = temporal_test_edges[0]
            ab = hypothesis_space - set(temporal_test_edges_no_tw)
            aupr_test_false_edges[k] = [(a, b, t, 1) for (a, b) in ab]

    # In the static aupr case, train/validation/split is not known in advance,
    #   so the aupr sets cannot be computed in advance.


    for split_num in range(0, num_splits):
        print("Working on split %d..." % split_num)
        if not temporal:  # static
            extension = "_%d.txt" % split_num

            train_edge_list = \
                sample(edges, train_size, with_replacement=False)

            remaining_edges = set(edges) - set(train_edge_list)
            test_true_edge_list = sample(remaining_edges, test_size, \
                with_replacement=False)

            valid_true_edge_list = \
                list(remaining_edges - set(test_true_edge_list))

            valid_false_edge_list = non_edges_sample(\
                nodes, edges, directed, valid_size, with_replacement=False)
            test_false_edge_list = non_edges_sample(\
                nodes, edges, directed, test_size, with_replacement=False)

        else:  # temporal
            extension = ".txt"

            # The true edges don't change. Only the false ones.
            train_edge_list = temporal_train_edges
            valid_true_edge_list = temporal_valid_edges
            test_true_edge_list = temporal_test_edges

            valid_timestamp = temporal_valid_edges[0][2]
            test_timestamp = temporal_test_edges[0][2]

            valid_false_edge_list = [(a, b, valid_timestamp, 1) for (a, b) in \
                    non_edges_sample(nodes, temporal_valid_edges_no_tw, \
                        directed, valid_size, with_replacement=False)]
            test_false_edge_list = [(a, b, test_timestamp, 1) for (a, b) in \
                    non_edges_sample(nodes, temporal_test_edges_no_tw, \
                        directed, test_size, with_replacement=False)]

        temporal_str = ["static", "temporal"][int(temporal)]
        directed_str = ["undirected", "directed"][int(directed)]
        base_path = "evaluation_data_splits/%s/%s_%s/" % \
            (temporal_str, graph_name, directed_str)
        # Export edges for ROC
        if split_num == 0:
            filepath = base_path + "node_list.txt"
            __export_items_list__(filepath, nodes, overwrite=overwrite)

        if (not temporal) or split_num == 0:
            filepath = base_path + "train_edges" + extension
            __export_items_list__(filepath, train_edge_list, overwrite=overwrite)
            filepath = base_path + "validation_true_edges" + extension
            __export_items_list__(filepath, valid_true_edge_list, \
                overwrite=overwrite)
            filepath = base_path + "test_true_edges" + extension
            __export_items_list__(filepath, test_true_edge_list, \
                overwrite=overwrite)
        filepath = base_path + "validation_false_edges_%d.txt" % split_num
        __export_items_list__(filepath, valid_false_edge_list, \
            overwrite=overwrite)
        filepath = base_path + "test_false_edges_%d.txt" % split_num
        __export_items_list__(filepath, test_false_edge_list, \
            overwrite=overwrite)

        # Now for AUPR
        if aupr:
            if temporal and split_num > 0:
                continue

            if not temporal:

                # Unlike in the temporal case, cannot be precomputed.

                # Create copies of the graph withholinding edges that are
                #   unknown after that evaluation (i.e. test edges in the case
                #   of validation).
                if directed:
                    g_valid_time = nx.DiGraph(graph)
                    g_test_time = nx.DiGraph(graph)
                else:
                    g_valid_time = nx.Graph(graph)
                    g_test_time = nx.Graph(graph)
                for (a, b) in test_true_edge_list:
                    g_valid_time.remove_edge(a, b)

            for k in range(k_for_aupr_range[0], k_for_aupr_range[1]):
                if temporal:
                    extension = ".txt"

                    # No sampling is needed.
                    train_edges = train_edge_list
                    valid_true_edges = aupr_valid_true_edges[k]
                    valid_false_edges = aupr_valid_false_edges[k]
                    test_true_edges = aupr_test_true_edges[k]
                    test_false_edges = aupr_test_false_edges[k]
                else:  # static
                    extension = "_%d.txt" % split_num

                    # Based on sampling from above, so ROC and AUPR are for the
                    #   same splits.
                    train_edges = train_edge_list

        # Strangely, during validation, we consider the test edges to be
        #   negatives (because they have not been added yet). Also, this
        #   choice prevents a method from learning the true test edges by noting
        #   which false edges are not included in the validation false edge set.

        # During testing, we consider the validation edges to have been added,
        #   so they do not count as negatives.

                    valid_false_edges = \
                        all_disconnected_node_pairs_within_k(g_valid_time,k)
                    test_false_edges = \
                        all_disconnected_node_pairs_within_k(g_test_time, k)

                    assert len(g_test_time.edges()) > 0

                    valid_allowed_true_edges = \
all_connected_node_pairs_that_would_be_within_k_if_disconnected(g_valid_time,k)
                    test_allowed_true_edges = \
all_connected_node_pairs_that_would_be_within_k_if_disconnected(g_test_time, k)

                    if not directed:
                        test_allowed_true_edges = \
                            set([(min(a,b), max(a,b)) for (a, b) in \
                                test_allowed_true_edges])
                        valid_allowed_true_edges = \
                            set([(min(a,b), max(a,b)) for (a, b) in \
                                valid_allowed_true_edges])

                    valid_true_edges = list(set(valid_true_edge_list) & \
                        valid_allowed_true_edges)
                    test_true_edges = list(set(test_true_edge_list) & \
                        test_allowed_true_edges)

                # Export edges for AUPR
                # if split_num == 0:
                #     filepath = base_path + "train_edges_k%d%s" % (k, extension)
                #     __export_items_list__(filepath, train_edges, \
                #         overwrite=overwrite)
                filepath = base_path + "validation_true_edges_k%d%s" % \
                    (k, extension)
                __export_items_list__(filepath, valid_true_edges, \
                    overwrite=overwrite)
                filepath = base_path + "validation_false_edges_k%d%s" % \
                    (k, extension)
                __export_items_list__(filepath, valid_false_edges, \
                    overwrite=overwrite)
                filepath = base_path + "test_true_edges_k%d%s" % \
                    (k, extension)
                __export_items_list__(filepath, test_true_edges, \
                    overwrite=overwrite)
                # False aupr edges are _always_ the same.
                if split_num == 0:
                    filepath = base_path + "test_false_edges_k%d.txt" % (k)
                    __export_items_list__(filepath, test_false_edges, \
                        overwrite=overwrite)

else:
    print("Note: evaluation_data_splits.py intended to be run as main.")
