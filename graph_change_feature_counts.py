import random
import math
from subgraph_change_labeler import SubgraphChangeLabeler
from graph_change import *
from multiprocessing import Pool
import numpy as np
import copy

class GraphChangeFeatureCounter:

    # __X__ denotes that variable or function X is for internal use only.
    #
    # `graph_data` -- the graph in which subgraphs will be labeled. Should be
    #       of type GraphData or DirectedGraphData from graph_data.py
    #
    # `apply_additions_while_labeling`
    #       If False, then at timestamp T:
    #           1. Apply all node additions.
    #           2. Label the changes.
    #           3. Apply all edge additions.
    #           4. Label the changes.
    #       If True, then at timestamp T:
    #           1. Apply and then immediately label additions one at a time.
    #
    # `apply_deletions_while_labeling`
    #       If False, then at timestamp T:
    #           1. Label all node and edge deletions.
    #           2. Apply all the changes.
    #       If True, then at timestamp T:
    #           1. Label and then immediately apply deletions one at a time.
    #
    # `subgraph_size` -- the size of the subgraphs that will be changing
    #
    # `node_traits` and `edge_traits` -- should be lists containing
    #       "rank traits" and/or "class traits"
    #       (from GraphChangeModeler.RankTrait() and .ClassTrait() below)
    #       Traits of the same names are expected to be in `graph_data`, but
    #       `graph_data` can have more traits than the labeler uses. The
    #       labeler ignores all traits not mentioned in the constructor.
    #
    # `node_trait_updaters` and `edge_trait_updaters` -- if used, should be
    #       lists of the same length as `node_traits` and `edge_traits`
    #       respectively. Each entry in the list should be an object of a
    #       subclass of TraitUpdater. Can be elements of the NonUpdater class
    #       which simply perform no updates.
    #
    # `differentiate_endpoints` -- affects labeling of edge changes -- the two
    #       nodes being (dis)connected will receive different labels. this is
    #       only relevant for undirected graphs, since directed edge changes
    #       already impose an ordering on the two nodes. Must be a function
    #       which, given a graph_data and two nodes, returns one of the nodes.
    #
    #       This function should NOT edit the graph_data, unless you are very
    #       confident in what you are doing.
    #
    #       For example:
    #           def diff(graph_data, node_a, node_b):
    #               d_a = len(graph_data.neighbors(node_a))
    #               d_b = len(graph_data.neighbors(node_b))
    #               if d_a < d_b:
    #                   return node_a
    #               if d_b < d_a:
    #                   return node_b
    #               ... some other criterion ...
    #
    # `deletion_reduced` -- whether or not node deletions should consider
    #       smaller subgraphs than the edge change and node addition cases.
    #       This is an option because node deletion is not limited to cases
    #       where the node has only a single edge (unlike node addition).
    #
    #       Note that, as of the writing of this comment, `deletion_reduced` is
    #       incompatible with precomputation.
    #
    # `precompute` -- used to signal whether the subgraph change labeler should
    #       precompute all possible labeling scenarios for faster labeling. Use
    #       with caution if adding traits or on large (especially directed)
    #       subgraphs. Valid values are True, False, or None. None lets the
    #       modeler automatically choose whether or not to precompute.
    #
    # `use_counts` -- rather than measuring distributions of subgraph->subgraph
    #   changes, use the raw counts.
    #
    # `num_processes` -- the number of processes you want the modeler to use.
    def __init__(self, graph_data, subgraph_size=4,
            apply_additions_while_labeling=True,
            apply_deletions_while_labeling=False,
            node_traits=[], edge_traits=[], \
            node_trait_updaters=[], edge_trait_updaters=[], \
            differentiate_endpoints=None, deletion_reduced=True,\
            precompute=False, use_counts=True, num_processes=4):

        print("Initializing GraphChangeFeatureCounter object.")

        if graph_data.is_directed() and differentiate_endpoints is not None:
            raise ValueError("Error! Will not differentiate endpoints on a " + \
                "directed graph. Set `differentiate_endpoints` to False.")

        if (not deletion_reduced) and precompute:
            raise ValueError("Error! Code currently does not support " + \
                "precomputation combined with full-size deletion. Either set "+\
                "`deletion_reduced` to True or `precompute` to False or None.")

        if len(node_trait_updaters) > 0 and \
                len(node_trait_updaters) != len(node_traits):
            raise ValueError("Error! `node_trait_updaters` of length > 0 but "+\
                "not of same length as `node_traits`.")

        if len(edge_trait_updaters) > 0 and \
                len(edge_trait_updaters) != len(edge_traits):
            raise ValueError("Error! `edge_trait_updaters` of length > 0 but "+\
                "not of same length as `edge_traits`.")

        self.__subgraph_size__ = subgraph_size
        self.__diff_ends__ = differentiate_endpoints
        self.__graph_data__ = graph_data
        self.__deletion_reduced__ = deletion_reduced

        self.__use_counts__ = use_counts

        self.__node_traits__ = node_traits
        self.__edge_traits__ = edge_traits
        self.__node_trait_updaters__ = node_trait_updaters
        self.__edge_trait_updaters__ = edge_trait_updaters

        self.__apply_additions__ = apply_additions_while_labeling
        self.__apply_deletions__ = apply_deletions_while_labeling

        self.__num_node_additions__ = 0
        self.__num_edge_additions__ = 0
        self.__num_edge_deletions__ = 0
        self.__num_node_deletions__ = 0

        self.__max_observed_labels__ = [-1, -1, -1, -1]

        self.__change_model_centroids__ = []
        self.__null_model_centroids__ = []

        self.__change_model_dists__ = []
        self.__null_model_dists__ = []

        self.__epsilon__ = 2.0**(-32)

        self.__num_processes__ = num_processes
        self.__pool__ = None

        self.__hashed_canonical_labels__ = None

        # self.change_counts_time = 0.0
        # self.other_time = 0.0

        if precompute is not None:
            self.__precompute__ = precompute
        else:
            # Set precompute automatically.
            et = len(self.__edge_traits__)
            nt = len(self.__node_traits__)
            self.__precompute__ = \
                (graph_data.is_directed() and \
                    ((et + nt == 0 and subgraph_size <= 5) or \
                     (et == 0 and nt == 1 and subgraph_size <= 4) or \
                     (et == 1 and nt == 0 and subgraph_size == 3))) or \
                ((not graph_data.is_directed()) and \
                    ((et + nt == 0 and subgraph_size <= 7) or \
                     (et <= 1 and nt <= 1 and subgraph_size <= 4) or \
                     (et == 0 and nt <= 2 and subgraph_size <= 3)))

        print("  Initializing SubgraphChangeLabeler for GraphChange" + \
            "FeatureCounter.")
        print("    (precompute set to %s)" % self.__precompute__)
        self.__change_labeler__ = \
            SubgraphChangeLabeler(graph_data, subgraph_size, \
                node_traits=self.__node_traits__, \
                edge_traits=self.__edge_traits__, \
                differentiate_endpoints=(self.__diff_ends__ is not None), \
                check_inputs=True, deletion_reduced=self.__deletion_reduced__, \
                precompute=self.__precompute__)
        print("  SubgraphChangeLabeler initialization complete.")

        print("GraphChangeFeatureCounter initialization complete.")

    def __del__(self):
        self.del_worker_pool()

    # If still has a worker pool, closes and deletes that pool.
    #
    # This class will create a new pool when it needs one.
    def del_worker_pool(self):
        if self.__pool__ is not None:
            self.__pool__.close()
            self.__pool__.join()
            self.__pool__ = None

    RANK_TRAIT = SubgraphChangeLabeler.RANK_TRAIT
    CLASS_TRAIT = SubgraphChangeLabeler.CLASS_TRAIT

    # A Rank Trait is a trait for nodes or edges.
    #
    # In the actual graph, this would be a number, but when converted to a
    #   subgraph, only the relative rankings of the values are used. If two
    #   nodes (or edges) have the same value, they will be given the same rank.
    #
    # If the values are guaranteed to be unique, set `guaranteed_unique` to
    #   true (affects counts of possible subgraphs).
    @staticmethod
    def RankTrait(name, guaranteed_unique=False):
        return (GraphChangeFeatureCounter.RANK_TRAIT, name, guaranteed_unique)

    # A Class Trait is a trait for nodes or edges.
    #
    # A Class Trait is like an ENUM: a variable can take on one value from a
    #   provided set of values, given in the `possible_class_values` list.
    @staticmethod
    def ClassTrait(name, possible_class_values):
        return (GraphChangeFeatureCounter.CLASS_TRAIT, name, \
            possible_class_values)

    # `changes` -- a list of changes -- should be elements of the GraphChange
    #       class -- the GraphChangeFeatureCounter will end up combining the
    #       changes in this list with those in `null_changes` and `non_changes`
    #       and sorting them according to the GraphChange class's __lt__
    #       method, so you should be fine with the changes being ordered
    #       according to a .sort() call.
    #
    #       Represents changes that actually occurred.
    #
    # `null_changes` -- same as `changes`, except that these changes should be
    #       changes that COULD have occurred. Note that this can overlap with
    #       changes that actually did occur (i.e. can contain elements that are
    #       also in `changes`.) These changes will not be applied to the
    #       `graph_data` the way that `changes` will.
    #
    # `non_changes` -- same as `changes`, except that these changes should be
    #       changes that COULD have occurred BUT DIDNT. Note that these changes
    #       can overlap with `non_changes` but _cannot_ overlap with `changes`.
    #       These changes will not be applied to the `graph_data` the way that
    #       `changes` will.
    #
    #       Note that using both `null_changes` and `non_changes` is potentially
    #       redundant since, IF the sets were full:
    #       `changes` UNION `non_changes` = `null_changes`
    #
    # `permanently_apply_changes` -- if true, the changes in `changes` will be
    #   performed on the GCFC's `graph_data` when the method is done.
    #
    # NOTE: `permanently_apply_changes` is not perfectly interfaced with trait
    #   updaters. What might work (unverified) is to call get_change_counts()
    #   twice - first with the null and non changes and `perm...` set to False,
    #   then second with the true changes and `perm...` set to True.
    #
    # `allow_new_SSTs` -- report counts for SSTs never seen before -- setting to
    #   false allows the algorithm a speedup when working with many distinct
    #   SSTs and many processes.
    def get_change_counts(self, changes, null_changes, non_changes,
            permanently_apply_changes=True, allow_new_SSTs=True):

        self.__create_worker_pool__()

        # The random.random() is to ensure that we don't measure all the null-
        #   or non-changes after the real changes are applied, but rather
        #   measure them after 

        changes = list(changes)
        for c in changes:
            c.set_permanent(permanently_apply_changes)
        null_changes = list(null_changes)
        for c in null_changes:
            c.set_permanent(False)
        non_changes = list(non_changes)
        for c in non_changes:
            c.set_permanent(False)

        change_list = [(c, random.random(), "C") for c in changes] + \
                      [(c, random.random(), "Null") for c in null_changes] + \
                      [(c, random.random(), "Non") for c in non_changes]
        change_list.sort()
        change_indices = set()
        null_indices = set()
        non_indices = set()
        for i in range(0, len(change_list)):
            (change, _, change_type) = change_list[i]
            if change_type == "C":
                change_indices.add(i)
            elif change_type == "Null":
                null_indices.add(i)
            else:
                non_indices.add(i)

        changes = [c for (c, _, _) in change_list]

        # Next lines only here to account for models pickled before
        #   these flags were in the code:
        if not hasattr(self.__change_labeler__, "__finished_labeling__"):
            self.__change_labeler__.__finished_labeling__ = False
        if not hasattr(self, "__hashed_canonical_labels__"):
            self.__hashed_canonical_labels__ = None

        if allow_new_SSTs:
            # If we allow new SSTs, destroy our old compressed version of
            #   canonical-subgraph-repr-to-label maps.
            self.__hashed_canonical_labels__ = None
        else:
            # If we haven't computed a low-memory way of telling threads how
            #   to re-label their count vectors, do so now.
            if self.__hashed_canonical_labels__ is None:
                self.__set_canon_shorthand__()

        index_ranges = self.__get_change_index_ranges__(changes)
        subgraph_to_subgraph_counts_dicts = \
            self.__parallel_get_counts__(\
                changes, index_ranges, apply_idxs_set=change_indices)

        if permanently_apply_changes:
            # Run the actual `graph_data` forward but do no measuring.
            self.__get_counts__(changes, index_ranges, measure_idxs_set=set(),
                apply_idxs_set=change_indices)

        change_counts_dicts = [[], [], [], []]
        null_counts_dicts = [[], [], [], []]
        non_counts_dicts = [[], [], [], []]
        for change_type_idx in range(0, 4):
            for (d, idx) in subgraph_to_subgraph_counts_dicts[change_type_idx]:
                if idx in change_indices:
                    change_counts_dicts[change_type_idx].append(d)
                elif idx in null_indices:
                    null_counts_dicts[change_type_idx].append(d)
                else:
                    non_counts_dicts[change_type_idx].append(d)

        return (change_counts_dicts, null_counts_dicts, non_counts_dicts)

    def get_max_seen_labels(self):
        return list(self.__max_observed_labels__)

    def get_subgraph_change_labeler(self):
        return self.__change_labeler__

    # Apply changes but don't count anything. Update traits as you go.
    def run_changes_forward(self, changes):
        changes = list(changes)
        for c in changes:
            c.set_permanent(True)
        index_ranges = self.__get_change_index_ranges__(changes)
        self.__get_counts__(changes, index_ranges, measure_idxs_set=set(),
            apply_idxs_set=set([i for i in range(0, len(changes))]))
        

    # If doesn't have a worker pool, creates one.
    def __create_worker_pool__(self):
        if self.__pool__ is None:
            self.__pool__ = Pool(self.__num_processes__)

    # Pools cannot pass themselves, so to pass the GCFC into self.__pool__, we
    #   need to remove the pool from our passed object.
    #
    # Also, pools have a bottleneck with passing arguments, so don't send an
    #   already-filled Subgraph Labeler.
    def __copy_for_parallel_calls__(self):
        c = copy.copy(self)
        c.__pool__ = None
        c.__change_labeler__ = \
            SubgraphChangeLabeler(self.__graph_data__, self.__subgraph_size__, \
                node_traits=self.__node_traits__, \
                edge_traits=self.__edge_traits__, \
                differentiate_endpoints=(self.__diff_ends__ is not None), \
                check_inputs=True, deletion_reduced=self.__deletion_reduced__, \
                precompute=False)
        return c

    def __set_up_change__(self, change, undo=False):
        if change.get_type() == GraphChange.NODE_ADDITION:
            change_type_idx = 0
        elif change.get_type() == GraphChange.EDGE_ADDITION:
            change_type_idx = 1
        elif change.get_type() == GraphChange.EDGE_DELETION:
            change_type_idx = 2
        elif change.get_type() == GraphChange.NODE_DELETION:
            change_type_idx = 3

        if undo:
            self.__graph_data__.set_savepoint()
        if change_type_idx == 0:
            if not self.__graph_data__.has_node(change.central_entity()):
                change.perform()
        if change_type_idx == 1:
            edge = change.central_entity()
            if not self.__graph_data__.has_edge(edge[0], edge[1]):
                change.perform()

        # Node trait updaters.
        for nt_updater in self.__node_trait_updaters__:
            nt_updater.update_just_before_change_labeled(change)
        # Edge trait updaters.
        for et_updater in self.__edge_trait_updaters__:
            et_updater.update_just_before_change_labeled(change)

        return change_type_idx

    def __values_dict_for_change__(self, change, undo=False):
        values = self.__get_change_counts__(change)
        if not self.__use_counts__:
            total = float(sum([c for l, c in values.items()]))
            values = {l: c / total for l, c in values.items()}

        if undo:
            self.__graph_data__.restore_to_savepoint()
            self.__graph_data__.clear_savepoint()
        return values

    def __surrounding_subgraphs__(self, nodes, target_size):
        nodes_set = set(nodes)
        num_additions = target_size - len(nodes)
        frontier = set()
        for node in nodes:
            for neighbor in self.__graph_data__.neighbors(node):
                if neighbor not in nodes_set and neighbor not in frontier:
                    frontier.add(neighbor)

        num_additions = target_size - len(nodes)

        off_limits = set(nodes)
        other_nodes_lists = []
        node_stack = []
        frontier_stack = [frontier]
        off_limits_stack = [off_limits]
        while len(frontier_stack) > 0:
            if len(node_stack) == num_additions:
                other_nodes_lists.append(list(node_stack))
                node_stack.pop()
                frontier_stack.pop()
                off_limits_stack.pop()
                continue

            old_frontier = frontier_stack[-1]
            if len(old_frontier) == 0:
                frontier_stack.pop()
                off_limits_stack.pop()
                if len(node_stack) > 0:
                    node_stack.pop()
                continue

            old_off_limits = off_limits_stack[-1]
            next_node = old_frontier.pop()
            old_off_limits.add(next_node)
            new_frontier = old_frontier | \
                (self.__graph_data__.neighbors(next_node) - old_off_limits)

            frontier_stack.append(new_frontier)
            off_limits_stack.append(set(old_off_limits))
            node_stack.append(next_node)

        return other_nodes_lists

    def __get_change_counts__(self, change):
        counts = {}
        entity = change.central_entity()

        if change.get_type() == GraphChange.NODE_ADDITION:
            change_type_idx = 0

            nodes = [entity] + list(self.__graph_data__.neighbors(entity))
            target_size = self.__subgraph_size__

            node = entity
            edge = (nodes[0], nodes[1])
            if self.__graph_data__.is_directed() and \
                    not self.__graph_data__.has_edge(edge[0], edge[1]):
                edge = (nodes[1], nodes[0])

            for other_nodes_list in \
                    self.__surrounding_subgraphs__(nodes, target_size):
                label = self.__change_labeler__.label_node_addition(\
                    node, edge, other_nodes_list)
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1

        elif change.get_type() == GraphChange.NODE_DELETION:
            change_type_idx = 3

            nodes = [entity]
            target_size = \
                self.__subgraph_size__ - int(self.__deletion_reduced__)

            node = entity
            for other_nodes_list in \
                    self.__surrounding_subgraphs__(nodes, target_size):
                label = self.__change_labeler__.label_node_deletion(\
                    node, other_nodes_list)
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1

        else: # EDGE_ADDITION or EDGE_DELETION
            nodes = list(entity)
            target_size = self.__subgraph_size__

            edge = entity

            if self.__diff_ends__ is not None:
                # Must call the __diff_ends__ function to see which order the
                #   nodes come in.
                first_node = \
                    self.__diff_ends__(self.__graph_data__, edge[0], edge[1])
                if first_node == edge[1]:
                    edge = (edge[1], edge[0])

            if change.get_type() == GraphChange.EDGE_ADDITION:
                change_type_idx = 1
                for other_nodes_list in \
                        self.__surrounding_subgraphs__(nodes, target_size):
                    label = self.__change_labeler__.label_edge_addition(\
                        edge, other_nodes_list)
                    if label not in counts:
                        counts[label] = 0
                    counts[label] += 1
            else:  # EDGE_DELETION
                change_type_idx = 2
                for other_nodes_list in \
                        self.__surrounding_subgraphs__(nodes, target_size):
                    label = self.__change_labeler__.label_edge_deletion(\
                        edge, other_nodes_list)
                    if label not in counts:
                        counts[label] = 0
                    counts[label] += 1


        for label, _ in counts.items():
            if label > self.__max_observed_labels__[change_type_idx]:
                self.__max_observed_labels__[change_type_idx] = label

        return counts

    def __get_change_index_ranges__(self, changes):
        index_ranges = []
        present_timestamp = changes[0].get_timestamp()
        present_change_type = None
        # Node-add start, edge-add start, edge-del start, node-del start, end.
        index_range = [[None, None, None, None], None]
        current_type_idx = -1
        for change_idx in range(0, len(changes)):
            change = changes[change_idx]
            c_timestamp = change.get_timestamp()
            c_type = change.get_type()
            if c_timestamp != present_timestamp:
                index_range[1] = change_idx
                for type_idx in range(current_type_idx + 1, 4):
                    index_range[0][type_idx] = change_idx
                index_ranges.append((tuple(index_range[0]), index_range[1]))
                index_range = [[None, None, None, None], None]
                present_change_type = None
                present_timestamp = c_timestamp
                current_type_idx = -1
            if c_type != present_change_type:
                if c_type == GraphChange.NODE_ADDITION:
                    new_type_idx = 0
                elif c_type == GraphChange.EDGE_ADDITION:
                    new_type_idx = 1
                elif c_type == GraphChange.EDGE_DELETION:
                    new_type_idx = 2
                else:
                    new_type_idx = 3
                for type_idx in range(current_type_idx + 1, new_type_idx + 1):
                    index_range[0][type_idx] = change_idx
                current_type_idx = new_type_idx

        index_range[1] = len(changes)
        for type_idx in range(current_type_idx + 1, 4):
            index_range[0][type_idx] = len(changes)
        index_ranges.append((tuple(index_range[0]), index_range[1]))

        return index_ranges

    def __parallel_index_split__(self, num_indices):
        target_num_splits = self.__num_processes__ #  * 3
        interval_size = int(num_indices / target_num_splits) + 1
        p = np.random.permutation([i for i in range(0, num_indices)])
        change_lists = [[p[j] for j in \
            range(i * interval_size, min((i+1) * interval_size, num_indices))] \
                for i in range(0, self.__num_processes__)]
        return [set(l) for l in change_lists]

    def __parallel_get_counts__(self, changes, index_ranges, apply_idxs_set):
        change_idxs_sets = \
            self.__parallel_index_split__(len(changes))
        new_GCFC = self.__copy_for_parallel_calls__()
        args = [(new_GCFC, changes, index_ranges, s, apply_idxs_set) \
                    for s in change_idxs_sets]
        result = self.__pool__.map(__parallel_model_counts_func__, args, chunksize=1)
        self.del_worker_pool()
        self.__create_worker_pool__()

        if self.__hashed_canonical_labels__ is None:
            # The results were each labeled by different subgraph change labelers,
            #   so their label-to-subgraph-change correspondences will be different.
            # Figure out what the labels should be, then send this back to the pool
            #   in order to relabel the data.
            label_overwrites = []
            for (alt_change_labeler, _) in result:

                overwrites_by_type = []
                for (cti, ct) in [(0, GraphChange.NODE_ADDITION), \
                        (1, GraphChange.EDGE_ADDITION), \
                        (2, GraphChange.EDGE_DELETION),\
                        (3, GraphChange.NODE_DELETION)]:
                    num_labels = \
                        alt_change_labeler.num_of_change_cases_for_type(ct)
                    overwrites = {}
                    for l in range(0, num_labels):
                        new_label = self.__change_labeler__.\
                            label_representative_subgraph_change(\
                                alt_change_labeler.\
    get_representative_subgraph_change_from_label(l, ct), ct)

                        if self.__max_observed_labels__[cti] < new_label:
                            self.__max_observed_labels__[cti] = new_label

                        overwrites[l] = new_label
                    overwrites_by_type.append(overwrites)
                label_overwrites.append(overwrites_by_type)

            # Now send the results to be re-labeled.
            args = [(label_overwrites[i], result[i][1]) for \
                i in range(0, len(result))]

            final_result = self.__pool__.map(__parallel_relabel_func__, args, chunksize=1)
            self.del_worker_pool()
            self.__create_worker_pool__()

        else:
            # The results were relabeled correctly via self.__hashed_canonical_labels__
            final_result = result

        change_counts_by_type = [[], [], [], []]
        for (change_counts) in final_result:
            for cti in range(0, 4):
                change_counts_by_type[cti] += change_counts[cti]

        return change_counts_by_type

    # Note: setting `measure_idxs_set` to None actually indicates that ALL
    #   should be measured. It's only by setting it to a set that you indicate
    #   that ONLY those in the set should be measured.
    def __get_counts__(self, changes, index_ranges, measure_idxs_set=None,
            apply_idxs_set=set()):

        change_counts_by_type = [[], [], [], []]

    # Refresher comments from specs:
    #
    # `apply_additions_while_labeling` -- (self.__apply_additions__)
    #       If False, then at timestamp T:
    #           1. Apply all node additions.
    #           2. Label the changes.
    #           3. Apply all edge additions.
    #           4. Label the changes.
    #       If True, then at timestamp T:
    #           1. Apply and then immediately label additions one at a time.
    #
    # `apply_deletions_while_labeling` -- (self.__apply_deletions__)
    #       If False, then at timestamp T:
    #           1. Label all node and edge deletions.
    #           2. Apply all the changes.
    #       If True, then at timestamp T:
    #           1. Label and then immediately apply deletions one at a time.

        overall_change_idx = -1

        for index_range in index_ranges:
            sub_indices = list(index_range[0]) + [index_range[1]]
            for change_type_idx in range(0, 4):
                change_range = [changes[idx] for idx in \
                    range(sub_indices[change_type_idx], \
                          sub_indices[change_type_idx + 1])]

                # If not self.__apply_additions__, apply _all_ node (or edge)
                #   additions before labeling the node (or edge) additions.
                if change_type_idx < 2 and not self.__apply_additions__:
                    for change in change_range:
                        change.perform()

                # Node trait updaters.
                for nt_updater in self.__node_trait_updaters__:
                    nt_updater.update_before_changes(change_range)
                # Edge trait updaters.
                for et_updater in self.__edge_trait_updaters__:
                    et_updater.update_before_changes(change_range)

                percent_done = 0
                for change_range_idx in range(0, len(change_range)):
                    overall_change_idx += 1

                    # `actually_measure` is used to differentiate between
                    #   changes for which we're collecting counts and changes
                    #   which are merely being executed.
                    actually_measure = True
                    if measure_idxs_set is not None and \
                            overall_change_idx not in measure_idxs_set:
                        actually_measure = False

                    if float(change_range_idx + 1) / len(change_range) > \
                            percent_done * .1:
                        percent_done += 1
                        # print("%d percent done" % (percent_done * 10))

                    use_real_change = overall_change_idx in apply_idxs_set

                    # If neither measuring nor executing, skip to next change.
                    if not actually_measure and not use_real_change:
                        continue

                    change = change_range[change_range_idx]

                    # Apply addition before labeling.
                    if use_real_change:
                        if change_type_idx < 2 and self.__apply_additions__:
                            change.perform()
                    else:  # For null- or non-change
                        self.__graph_data__.set_savepoint(\
                            name="GraphChangeFeatureCounter Savepoint")
                        if change_type_idx < 2:
                            change.perform()

                    # Node trait updaters.
                    for nt_updater in self.__node_trait_updaters__:
                        nt_updater.update_just_before_change_labeled(change)
                    # Edge trait updaters.
                    for et_updater in self.__edge_trait_updaters__:
                        et_updater.update_just_before_change_labeled(change)

                    if actually_measure:
                        counts = self.__get_change_counts__(change)

                    if actually_measure:
                        change_counts_by_type[change_type_idx].append(\
                            (counts, overall_change_idx))

                    # If deletion to be applied, do so after labeling.
                    if use_real_change and change_type_idx >= 2 and \
                            self.__apply_deletions__:
                        change.perform()

                    if use_real_change:
                        # Node trait updaters.
                        for nt_updater in self.__node_trait_updaters__:
                            nt_updater.\
                                update_just_after_change_labeled(change)
                        # Edge trait updaters.
                        for et_updater in self.__edge_trait_updaters__:
                            et_updater.\
                                update_just_after_change_labeled(change)
                    else:  # A non- or null-change.
                        # Revert changes.
                        self.__graph_data__.restore_to_savepoint()
                        self.__graph_data__.clear_savepoint()

                # If not self.__apply_deletions__, wait to perform _any_
                #   deletions until both edge and node deletions have been
                #   labeled.
                if change_type_idx == 3 and not self.__apply_deletions__:
                    overall_idx = sub_indices[2]
                    while overall_idx < sub_indices[4]:
                        changes[overall_idx].perform()
                        overall_idx += 1

                # Node trait updaters.
                for nt_updater in self.__node_trait_updaters__:
                    nt_updater.update_after_changes(change_range)
                # Edge trait updaters.
                for et_updater in self.__edge_trait_updaters__:
                    et_updater.update_after_changes(change_range)


        return change_counts_by_type

    # Takes all the (label, canonical representation) pairs and compresses
    #   this info with collision-free hashing so that the worker pool's
    #   processes can return the correct labels without transimitting their
    #   entire SST collection.
    def __set_canon_shorthand__(self):
        collisions = [{}, {}, {}, {}]
        hash_val_to_label = [{}, {}, {}, {}]
        for (cti, ct) in [(0, GraphChange.NODE_ADDITION), \
                          (1, GraphChange.EDGE_ADDITION), \
                          (2, GraphChange.EDGE_DELETION), \
                          (3, GraphChange.NODE_DELETION)]:
            for l in range(0, self.__change_labeler__.num_of_change_cases_for_type(ct)):
                canonical_repr = self.__change_labeler__.\
get_representative_subgraph_change_from_label(l, ct)
                canonical_hash = hash(canonical_repr)
                if canonical_hash in collisions[cti]:
                    collisions[cti][canonical_hash].add((l, canonical_repr))
                else:
                    collisions[cti][canonical_hash] = set([(l, canonical_repr)])
            for (canonical_hash, s) in collisions[cti].items():
                if len(s) > 1:
                    for (l, canonical_repr) in s:
                        hash_val_to_label[cti][canonical_repr] = l
                else:
                    for (l, canonical_repr) in s:
                        hash_val_to_label[cti][canonical_hash] = l
        collisions = None
        self.__hashed_canonical_labels__ = hash_val_to_label

def __parallel_model_counts_func__(arg):
    (GCFC, changes, index_ranges, measure_idxs_set, apply_idxs_set) = arg

    if GCFC.__hashed_canonical_labels__ is None:
        counts = GCFC.__get_counts__(changes, index_ranges,
            measure_idxs_set=measure_idxs_set, \
            apply_idxs_set=apply_idxs_set)
        GCFC.__change_labeler__.finish_labeling()
        return (GCFC.__change_labeler__, counts)
    else:
        counts = GCFC.__get_counts__(changes, index_ranges,
            measure_idxs_set=measure_idxs_set, \
            apply_idxs_set=apply_idxs_set)
        change_labeler = GCFC.get_subgraph_change_labeler()

        # Create relabeling maps that will map labels here to the parent
        #   process's labels.
        relabel_maps = [{}, {}, {}, {}]
        for (cti, ct) in [(0, GraphChange.NODE_ADDITION), \
                          (1, GraphChange.EDGE_ADDITION), \
                          (2, GraphChange.EDGE_DELETION), \
                          (3, GraphChange.NODE_DELETION)]:
            for l in range(0, change_labeler.num_of_change_cases_for_type(ct)):
                canonical_repr = change_labeler.\
get_representative_subgraph_change_from_label(l, ct)
                canonical_hash = hash(canonical_repr)
                if canonical_hash in GCFC.__hashed_canonical_labels__[cti]:
                    relabel_maps[cti][l] = \
                        GCFC.__hashed_canonical_labels__[cti][canonical_hash]
                elif canonical_repr in GCFC.__hashed_canonical_labels__[cti]:
                    relabel_maps[cti][l] = \
                        GCFC.__hashed_canonical_labels__[cti][canonical_repr]

        # Use relabeling maps.
        for cti in range(0, 4):
            for idx in range(0, len(counts[cti])):
                relabeled_counts_dict = {}
                (old_counts_dict, change_idx) = counts[cti][idx]
                for old_label, c in old_counts_dict.items():
                    if old_label in relabel_maps[cti]:
                        relabeled_counts_dict[relabel_maps[cti][old_label]] = c
                counts[cti][idx] = (relabeled_counts_dict, change_idx)

        return (counts)
                    

def __parallel_relabel_func__(arg):
    (label_overwrites, labels) = arg
    new_labels = [[], [], [], []]
    for cti in range(0, 4):
        new_labels[cti] = [({label_overwrites[cti][l]: c for l,c in d.items()},\
            parallel_idx) for (d, parallel_idx) in labels[cti]]

    return new_labels

if __name__ == "__main__":
    from graph_data import GraphData, DirectedGraphData

    gd = DirectedGraphData()
    for i in range(0, 10):
        gd.add_node(i)

    for i in range(0, 6):
        for j in range(i + 1, 6):
            gd.add_edge(i, j)

    gd.add_edge(0, 6)
    gd.add_edge(6, 7)
    gd.add_edge(7, 8)
    gd.add_edge(8, 9)
    gd.add_edge(9, 1)

    changes = [NodeAddition(gd, 10, 0, source=10), EdgeAddition(gd, 0, 8), EdgeAddition(gd, 0, 7), EdgeDeletion(gd, 0, 3), NodeDeletion(gd, 4)]

    GCFC = GraphChangeFeatureCounter(gd, subgraph_size=4, use_counts=True, precompute=False)
    (counts, _, _) = GCFC.get_change_counts(changes, [], [])
    print(counts[0])
    print(counts[1])
    print(counts[2])
    print(counts[3])
