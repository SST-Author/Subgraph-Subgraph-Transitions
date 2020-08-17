from degree_trait import *
import gc
from graph_change import GraphChange, EdgeAddition
from graph_change_feature_counts import GraphChangeFeatureCounter
from graph_data import GraphData, DirectedGraphData
from link_predictor import StaticLinkPredictor, TemporalLinkPredictor
from link_sample_utils import *
import math
from multiprocessing import Pool
import networkx as nx
import numpy as np
import random
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sys
from temporal_link_pred_traits import TemporalLinkPredictionTraits, TemporalLinkPredictionTraitUpdater
from trait_updater import NonUpdater

# SST_SVMTemporalLinkPredictor defined below in a similar but distinct fashion.

class SST_SVMLinkPredictor(StaticLinkPredictor):

    # `prediction_dist_cap` -- used to indicate that the predictor will only
    #   be used to make predictions about connecting pairs of nodes at most
    #   a distance of `prediction_dist_cap` away. A value of None indicates no
    #   limit.
    def __init__(self, graph_nodes, graph_edges, directed=False, \
            subgraph_size=4, non_edge_multiplier=10, \
            prediction_dist_cap=None, \
            num_processes=8, scale_data=False):

        self.__scale_data__ = scale_data

        if directed:
            self.__graph_data__ = DirectedGraphData()
        else:
            self.__graph_data__ = GraphData()
        for node in graph_nodes:
            self.__graph_data__.add_node(node)

        for (a, b) in graph_edges: # remaining_edges
            self.__graph_data__.add_edge(a, b)

        num_nodes = len(graph_nodes)
        num_edges = len(graph_edges)
        num_non_edges = int((num_nodes * (num_nodes - 1)) / \
            (2 - int(directed))) - num_edges
        
        if prediction_dist_cap is None:
            true_edges = graph_edges

            target_non_edge_size = min(num_non_edges, \
                                       len(true_edges) * non_edge_multiplier)
            non_edges = non_edges_sample(graph_nodes, graph_edges, directed,
                target_non_edge_size, with_replacement=False)
        else:
            k = prediction_dist_cap
            true_edges = \
                all_connected_node_pairs_that_would_be_within_k_if_disconnected(\
                    self.__graph_data__, k)

            possible_edges = \
                all_disconnected_node_pairs_within_k(self.__graph_data__, k)

            target_non_edge_size = min(len(possible_edges), \
                                       len(true_edges) * non_edge_multiplier)

            non_edges = set(random.sample(possible_edges, \
                target_non_edge_size))

        print("Training on %d true edges (%f percent of all graph edges)" % \
            (len(true_edges), (100.0 * len(true_edges)) / len(graph_edges)))

        print("Training on %d non model edges (%f percent of all non edges.)" % \
            (len(non_edges), (100.0 * len(non_edges)) / num_non_edges))

        true_changes = self.__edges_to_changes__(true_edges)
        non_changes = self.__edges_to_changes__(non_edges)

        if self.__graph_data__.is_directed():
            node_traits = []
            node_trait_updaters = []
        else:
            node_traits = [InvolvedNodeDegreeTrait()]
            node_trait_updaters = \
                [InvolvedNodeDegreeTraitUpdater(self.__graph_data__)]

        self.__GCFC__ = GraphChangeFeatureCounter(self.__graph_data__, \
            num_processes=num_processes, subgraph_size=subgraph_size, \
            node_traits=node_traits, node_trait_updaters=node_trait_updaters, \
            use_counts=True)

        self.__true_dicts__, _, self.__non_dicts__ = \
            self.__GCFC__.get_change_counts(true_changes, [], non_changes)

        # Get the edge additions specifically.
        self.__true_dicts__ = self.__true_dicts__[1]
        self.__non_dicts__ = self.__non_dicts__[1]

        print("Finished training data counting.")

    def score_edges(self, edges):
        changes = self.__edges_to_changes__(edges)
        # Perform scoring in chunks to save memory.
        scores = []
        chunk = 0
        chunk_size = 6000
        done = False
        stop = 0
        while not done:
            chunk += 1
            start = stop
            if chunk * chunk_size >= len(changes):
                stop = len(changes)
                done = True
            else:
                stop = chunk * chunk_size
            changes_to_score = changes[start:stop]
            scores += self.score_changes(changes_to_score)

            gc.collect()
            print("  Scored chunk %d." % chunk)
        return scores

    def score_changes(self, changes):
        # Pass as null_changes so that self's graph_data doesn't change.
        print("  Getting Changes' SST Vectors...")
        _, counts_dicts, _ = self.__GCFC__.get_change_counts([], changes, [], \
            permanently_apply_changes=False, allow_new_SSTs=False)
        print("  Scoring...")
        # Convert in place to save space.
        if self.__scale_data__:
            self.__scale_dicts__(counts_dicts[1])
        counts_vectors = self.__dicts_to_sparse_matrix__(counts_dicts[1])
        return self.score_vectors(counts_vectors)
        
    def score_vectors(self, count_vectors):
        return list(self.__linear_svm__.decision_function(count_vectors))

    # def graph(self):
    #     return self.__graph_data__

    # Returns the unit direction vector with components sorted in order of
    #   largest magnitude to least, coupled with a representation of the
    #   subgraph changes associated with each component.
    #
    # Format: List of (vector component, representative subgraph change) tuples
    def get_interpretable_model(self):
        # Extract interpretable features.
        direction_vector = self.__linear_svm__.coef_[0]
        norm = math.sqrt(sum([c*c for c in direction_vector]))
        direction_vector = [c / norm for c in direction_vector]
        sst_labeler = self.__GCFC__.get_subgraph_change_labeler()
        ssts = [sst_labeler.get_representative_subgraph_change_from_label(i, \
            GraphChange.EDGE_ADDITION) for i in range(0, len(direction_vector))]

        dv_sorted = [(abs(direction_vector[i]), direction_vector[i], i) \
            for i in range(0, len(direction_vector))]
        dv_sorted.sort(reverse=True)

        return [(dv_sorted[i][1], ssts[dv_sorted[i][2]]) \
            for i in range(0, len(ssts))]

    # Allows python to pickle the predictor.
    #
    # Once the predictor is used to make a prediction, this method will need to
    #   be called again in order for pickling to work.
    def become_serializeable(self):
        self.__GCFC__.del_worker_pool()

    def __del__(self):
        del self.__GCFC__

    def fit(self):

        self.__num_labels__ = self.__GCFC__.get_max_seen_labels()[1] + 1

        # Save space with sparse row matrix.
        # Construct while deleting dicts so it's effectively in place.
        num_true = len(self.__true_dicts__)
        num_non = len(self.__non_dicts__)
        all_dicts = self.__true_dicts__
        for i in range(0, num_non):
            all_dicts.append(self.__non_dicts__.pop())

        if self.__scale_data__:
            self.__feature_maxs__ = [1.0 for i in range(0, self.__num_labels__)]
            for d in all_dicts:
                for label, count in d.items():
                    if float(count) > self.__feature_maxs__[label]:
                        self.__feature_maxs__[label] = float(count)
            self.__scale_dicts__(all_dicts)
        
        data_matrix = self.__dicts_to_sparse_matrix__(all_dicts)

        self.__true_dicts__ = None
        self.__non_dicts__ = None

        self.__linear_svm__ = LinearSVC(class_weight='balanced', max_iter=400000)
        # non labels come first because __dicts_to_sparse_matrix__ reverses
        #   row order.
        labels = [0 for i in range(0, num_non)] + \
            [1 for i in range(0, num_true)]

        print("  Now fitting SVM...")

        self.__linear_svm__.fit(data_matrix, labels)

        data_matrix = None
        gc.collect()

        print("  SVM fit successfully.")

    # Destroys dicts in the process.
    def __dicts_to_sparse_matrix__(self, dicts):
        data = []
        row_idxs = []
        col_idxs = []
        size = len(dicts)
        for row in range(0, size):
            row_dict = dicts.pop()
            for col, value in row_dict.items():
                if col >= self.__num_labels__:
                    continue
                data.append(value)
                row_idxs.append(row)
                col_idxs.append(col)
        return csr_matrix((data, (row_idxs, col_idxs)), \
            shape=(size, self.__num_labels__))

    def __edges_to_changes__(self, edges):
        changes = []
        for (a, b) in edges:
            changes.append(EdgeAddition(self.__graph_data__, a, b))
        return changes

    def __scale_dicts__(self, dicts):
        for d in dicts:
            vals = [(label, float(count)) for (label, count) in d.items()]
            for (label, count) in vals:
                if label < self.__num_labels__:
                    d[label] = count / self.__feature_maxs__[label]

class SST_SVMTemporalLinkPredictor(TemporalLinkPredictor):

    # `non_edge_multiplier` - for every true edge, sample this many false edges
    #
    # `base_frac` - have at least this fraction of edges in the graph before
    #   computing vectors for the subsequent edges. Will make split at a
    #   timestamp change, and thus will always have at least the first full
    #   timestamp in the base_graph, even if base_frac=0.0. Also, will always
    #   have at least the last full timestamp outside the base graph, even if
    #   base_frac=1.0
    def __init__(self, graph_nodes, graph_edges, directed=False, \
            subgraph_size=4, non_edge_multiplier=10, \
            num_processes=8, base_frac=1.0, scale_data=False):

        self.__scale_data__ = scale_data

        if directed:
            self.__graph_data__ = DirectedGraphData()
        else:
            self.__graph_data__ = GraphData()

        for node in graph_nodes:
            self.__graph_data__.add_node(node)

        traits = TemporalLinkPredictionTraits
        # NonUpdater included for reasons explained in
        #   TemporalLinkPredictionTraitUpdater's file and the
        #   GraphChangeFeatureCounter file.
        #
        # In short, GCFC needs two updaters, but both the temporal traits used
        #   here operate with a single updater.
        trait_updaters = [\
          TemporalLinkPredictionTraitUpdater(self.__graph_data__), \
            NonUpdater(None)]

        # Remove weight value (this class ignores it) and sort by time.
        sorted_edges = [(a, b, t) for (t, a, b) in \
            sorted([(t, a, b) for (a, b, t, w) in graph_edges])]

        # Pick the timestamp from `base_frac` of the way through the data,
        #   then add all edges with a timestamp <= to it and allow the traits 
        #   to update accordingly. This will be the base graph. But first,
        #   ensure that the base graph does not include the last timestamp.
        last_timestamp = sorted_edges[-1][2]
        base_graph_timestamp_idx = min(len(sorted_edges) - 1, \
                                       int(len(sorted_edges)*base_frac))
        base_graph_timestamp = sorted_edges[base_graph_timestamp_idx][2]
        
        while base_graph_timestamp == last_timestamp:
            base_graph_timestamp_idx -= 1
            base_graph_timestamp = sorted_edges[base_graph_timestamp_idx][2]

        changes = []

        self.__GCFC__ = GraphChangeFeatureCounter(self.__graph_data__, \
            num_processes=num_processes, subgraph_size=subgraph_size, \
            edge_traits=traits, edge_trait_updaters=trait_updaters, \
            use_counts=True)

        for (a, b, t) in sorted_edges:
            if t > base_graph_timestamp:
                break
            changes.append(EdgeAddition(self.__graph_data__, a, b, timestamp=t))
        self.__GCFC__.run_changes_forward(changes)

        # Create fake edges for remaining timestamps in graph.
        curr_idx = 0
        while sorted_edges[curr_idx][2] <= base_graph_timestamp:
            curr_idx += 1

        print(("Used first %d edges for base graph. " % curr_idx) + \
            "Using remaining %d for change model." % \
                (len(sorted_edges) - curr_idx))

        start_idx = curr_idx
        curr_time = sorted_edges[curr_idx][2]

        num_nodes = len(graph_nodes)

        self.__true_dicts__ = []
        self.__non_dicts__ = []

        edges_at_curr_time = []
        end = False
        while not end:
            if curr_idx < len(sorted_edges):
                (a, b, t) = sorted_edges[curr_idx]
            else:
                end = True
            if end or t > curr_time:
                num_edges = len(edges_at_curr_time)

                num_non_edges = int((num_nodes * (num_nodes - 1)) / \
                    (2 - int(directed))) - num_edges
                target_non_edge_size = min(num_non_edges, \
                    len(edges_at_curr_time) * non_edge_multiplier)

                non_edges = non_edges_sample(graph_nodes, \
                    [(u, v) for (u, v, t) in edges_at_curr_time], \
                    directed, target_non_edge_size, with_replacement=False)

                fake_edges = [(u, v, curr_time) for (u, v) in non_edges]

                true_changes = self.__edges_to_changes__(edges_at_curr_time)
                non_changes = self.__edges_to_changes__(fake_edges)
                # Pass true changes as null changes to they don't accumulate
                #   during this timestep.
                _, true_dicts, non_dicts = \
                    self.__GCFC__.get_change_counts([], true_changes, \
                        non_changes, \
                        permanently_apply_changes=False)

                # Then run changes forward.
                self.__GCFC__.run_changes_forward(true_changes)

                # Get the edge additions specifically.
                self.__true_dicts__ += true_dicts[1]
                self.__non_dicts__ += non_dicts[1]

                curr_time = t
                edges_at_curr_time = []

            edges_at_curr_time.append((a, b, t))
            curr_idx += 1

        print("Finished training data counting.")

    def score_edges(self, edges):
        changes = self.__edges_to_changes__(edges)
        # Perform scoring in chunks to save memory.
        scores = []
        chunk = 0
        chunk_size = 12000
        done = False
        stop = 0
        while not done:
            chunk += 1
            start = stop
            if chunk * chunk_size >= len(changes):
                stop = len(changes)
                done = True
            else:
                stop = chunk * chunk_size
            changes_to_score = changes[start:stop]
            scores += self.score_changes(changes_to_score)

            gc.collect()
            print("  Scored chunk %d." % chunk)
        return scores

    def score_changes(self, changes):
        # Pass as null_changes so that self's graph_data doesn't change.
        print("  Getting Changes' SST Vectors...")
        _, counts_dicts, _ = self.__GCFC__.get_change_counts([], changes, [], \
            permanently_apply_changes=False, allow_new_SSTs=False)
        print("  Scoring...")
        # Convert in place to save space.
        if self.__scale_data__:
            self.__scale_dicts__(counts_dicts[1])
        counts_vectors = self.__dicts_to_sparse_matrix__(counts_dicts[1])
        return self.score_vectors(counts_vectors)
        
    def score_vectors(self, count_vectors):
        return list(self.__linear_svm__.decision_function(count_vectors))

    # Returns the unit direction vector with components sorted in order of
    #   largest magnitude to least, coupled with a representation of the
    #   subgraph changes associated with each component.
    #
    # Format: List of (vector component, representative subgraph change) tuples
    def get_interpretable_model(self):
        # Extract interpretable features.
        direction_vector = self.__linear_svm__.coef_[0]
        norm = math.sqrt(sum([c*c for c in direction_vector]))
        direction_vector = [c / norm for c in direction_vector]
        sst_labeler = self.__GCFC__.get_subgraph_change_labeler()
        ssts = [sst_labeler.get_representative_subgraph_change_from_label(i, \
            GraphChange.EDGE_ADDITION) for i in range(0, len(direction_vector))]

        dv_sorted = [(abs(direction_vector[i]), direction_vector[i], i) \
            for i in range(0, len(direction_vector))]
        dv_sorted.sort(reverse=True)

        return [(dv_sorted[i][1], ssts[dv_sorted[i][2]]) \
            for i in range(0, len(ssts))]

    # Allows python to pickle the predictor.
    #
    # Once the predictor is used to make a prediction, this method will need to
    #   be called again in order for pickling to work.
    def become_serializeable(self):
        self.__GCFC__.del_worker_pool()

    def fit(self):

        self.__num_labels__ = self.__GCFC__.get_max_seen_labels()[1] + 1

        # Save space with sparse row matrix.
        # Construct while deleting dicts so it's effectively in place.
        num_true = len(self.__true_dicts__)
        num_non = len(self.__non_dicts__)
        all_dicts = self.__true_dicts__
        for i in range(0, num_non):
            all_dicts.append(self.__non_dicts__.pop())

        if self.__scale_data__:
            self.__feature_maxs__ = [1.0 for i in range(0, self.__num_labels__)]
            for d in all_dicts:
                for label, count in d.items():
                    if float(count) > self.__feature_maxs__[label]:
                        self.__feature_maxs__[label] = float(count)
            self.__scale_dicts__(all_dicts)

        data_matrix = self.__dicts_to_sparse_matrix__(all_dicts)

        self.__true_dicts__ = None
        self.__non_dicts__ = None

        self.__linear_svm__ = LinearSVC(class_weight='balanced', max_iter=400000)
        # non labels come first because __dicts_to_sparse_matrix__ reverses
        #   row order.
        labels = [0 for i in range(0, num_non)] + \
            [1 for i in range(0, num_true)]
        print("  Now fitting SVM...")

        self.__linear_svm__.fit(data_matrix, labels)

        data_matrix = None
        gc.collect()

        print("  SVM fit successfully.")

    def __del__(self):
        del self.__GCFC__

    # Destroys dicts in the process.
    def __dicts_to_sparse_matrix__(self, dicts):
        data = []
        row_idxs = []
        col_idxs = []
        size = len(dicts)
        for row in range(0, size):
            row_dict = dicts.pop()
            for col, value in row_dict.items():
                if col >= self.__num_labels__:
                    continue
                data.append(value)
                row_idxs.append(row)
                col_idxs.append(col)
        return csr_matrix((data, (row_idxs, col_idxs)), \
            shape=(size, self.__num_labels__))

    def __edges_to_changes__(self, edges):
        changes = []
        for (a, b, t) in edges:
            changes.append(EdgeAddition(self.__graph_data__, a, b, timestamp=t))
        return changes

    def __scale_dicts__(self, dicts):
        for d in dicts:
            vals = [(label, float(count)) for (label, count) in d.items()]
            for (label, count) in vals:
                if label < self.__num_labels__:
                    d[label] = count / self.__feature_maxs__[label]
