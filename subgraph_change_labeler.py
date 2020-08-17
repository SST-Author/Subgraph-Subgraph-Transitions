from graph_change import GraphChange
from graph_data import GraphData, DirectedGraphData
import algorithmic_utils as a_utils
import graph_utils as g_utils

"""
The SubgraphChangeLabeler class does the following:

*  Label subgraphs _with respect to_ a stated change occuring in that subgraph.
   For example, label subgraph G1 with respect to an edge being added between
     its first and second nodes.
   Essentially, the resultant label is a label for the pair of:
     (subgraph as it is, new subgraph after change occurs)
   Given two such pairs, (A, B), (C, D) the labeler gives the pairs the same
     label if and only if A is isomorphic to C and B is isomorphic to D.
   The _input_ to the labeler is not actually two subgraphs but rather the
     first subgraph coupled with edit information (e.g. "add edge between nodes
     x1 and x2").

The labeler works for directed and undirected graphs, and allows for node and
edge properties ("traits").
"""

class SubgraphChangeLabeler:

    # __X__ denotes that variable or function X is for internal use only.

    # `graph_data` -- the graph in which subgraphs will be labeled. Should be
    #       of type GraphData or DirectedGraphData from graph_data.py
    #
    # `subgraph_size` -- the size of the subgraphs that will be changing
    #
    # `node_traits` and `edge_traits` -- should be lists containing
    #       "rank traits" and/or "class traits"
    #       (from GraphChangeModeler.RankTrait() and .ClassTrait())
    #       Traits of the same names are expected to be in `graph_data`, but
    #       `graph_data` can have more traits than the labeler uses. The
    #       labeler ignores all traits not mentioned in the constructor.
    #
    # `differentiate_endpoints` -- affects labeling of edge changes -- the two
    #       nodes being (dis)connected will receive different labels. this is
    #       only relevant for undirected graphs, since directed edge changes
    #       already impose an ordering on the two nodes.
    #
    # `check_inputs` -- purely for diagnostic purposes, signals to check that
    #       some inputs are as expected. Disable for code speed.
    #
    # `deletion_reduced` -- whether or not node deletions should consider
    #       smaller subgraphs than the edge change and node addition cases.
    #       This is an option because node deletion is not limited to cases
    #       where the node has only a single edge (unlike node addition).
    #
    #       NOTE: Code presently does not support setting `deletion_reduced` to
    #           False if precompute is selected.
    #
    # `precompute` -- whether or not the labeler should precompute all possible
    #       inputs so that it can more quickly label subgraphs. Note that if
    #       precompute is not enabled, the class will not know the number of
    #       total possible labels; it would only know the number of labels (i.e.
    #       distinct subgraph->subgraph changes) it has seen thus far.
    #
    #       NOTE: precompute is presently broken and should not be used.
    def __init__(self, graph_data, subgraph_size, \
            node_traits=[], edge_traits=[], differentiate_endpoints=False, \
            check_inputs=True, deletion_reduced=True, precompute=True):

        if check_inputs and precompute and subgraph_size < 3:
            raise ValueError("Error! Code requires subgraphs of size >= 3.")

        if graph_data.is_directed() and differentiate_endpoints:
            raise ValueError("Error! Will not differentiate endpoints on a " + \
                "directed graph. Set `differentiate_endpoints` to False.")

        if (not deletion_reduced) and precompute:
            raise ValueError("Error! Code currently does not support " + \
                "precomputation combined with full-size deletion. Either " + \
                "set `deletion_reduced` to True or `precompute` to False.")

        self.__subgraph_size__ = subgraph_size
        self.__diff_ends__ = differentiate_endpoints
        self.__graph_data__ = graph_data
        self.__node_traits__ = node_traits
        self.__edge_traits__ = edge_traits
        self.__check_inputs__ = check_inputs
        self.__deletion_modifier__ = 0 - int(deletion_reduced)
        self.__precompute__ = precompute
        self.__finished_labeling__ = False

        # Allows skipping over isomorphism concerns later for fast labeling.
        if self.__precompute__:
            self.__enumerate_and_label_all_possible_inputs__()
            return

        # Otherwise, no precomputation.

        # Note that these values are also set in __enumerate_and_label...()
        self.__DEL_NODE__ = 0
        self.__ADD_NODE__ = 1
        self.__DEL_EDGE__ = 2
        self.__ADD_EDGE__ = 3
        self.__num_labels__ = {0: 0, 1: 0, 2: 0, 3: 0}
        self.__repr_to_label__ = {0: dict(), 1: dict(), 2: dict(), 3: dict()}
        self.__label_to_canonical_repr__ = \
            {0: dict(), 1: dict(), 2: dict(), 3: dict()}

    RANK_TRAIT = 0
    CLASS_TRAIT = 1

    # Input:
    #   `edge` -- a pair of nodes in the subgraph to be connected
    #   `other_nodes` -- a list of the nodes forming the subgraph that are NOT
    #       mentioned in `edge`
    #
    # Output:
    #    (automorphism-invariant) label of the edge change
    def label_edge_addition(self, edge, other_nodes):
        if self.__finished_labeling__:
            raise ValueError("Error! Cannot call label_edge_addition() " + \
                "after finish_labeling() is called.")

        if self.__check_inputs__:
            if len(other_nodes) != self.__subgraph_size__ - 2:
                raise ValueError("Error! Expected `other_nodes` to be" +\
                    " of size %d, was %d." % \
                    (self.__subgraph_size__ - 2, len(other_nodes)))
            if not self.__graph_data__.has_edge(edge[0], edge[1]):
                raise ValueError("Error! graph_data not have edge " + \
                    str(edge) + "(Additions are required to be present in " + \
                    "graph in order to label them.).")

        return self.__label_edge_change__(edge, other_nodes, self.__ADD_EDGE__)

    # Input:
    #   `edge` -- a pair of nodes in the subgraph to be disconnected
    #   `other_nodes` -- a list of the nodes forming the subgraph that are NOT
    #       mentioned in `edge`
    #
    # Output:
    #    (automorphism-invariant) label of the edge change
    def label_edge_deletion(self, edge, other_nodes):
        if self.__finished_labeling__:
            raise ValueError("Error! Cannot call label_edge_deletion() " + \
                "after finish_labeling() is called.")

        if self.__check_inputs__:
            if len(other_nodes) != self.__subgraph_size__ - 2:
                raise ValueError("Error! Expected `other_nodes` to be" +\
                    " of size %d, was %d." % \
                    (self.__subgraph_size__ - 2, len(other_nodes)))
            if not self.__graph_data__.has_edge(edge[0], edge[1]):
                raise ValueError("Error! graph_data does not have edge " + \
                    str(edge))

        return self.__label_edge_change__(edge, other_nodes, self.__DEL_EDGE__)

    # Input:
    #   `node` -- the node in the subgraph that is to be added -- note that
    #       `node` must be a node in the graph_data
    #   `edge` -- the connection by which `node` will join
    #   `other_nodes` -- a list of the nodes forming the subgraph that are NOT
    #       mentioned in `edge`
    #
    # Output:
    #    (automorphism-invariant) label of the node addition
    def label_node_addition(self, node, edge, other_nodes):
        if self.__finished_labeling__:
            raise ValueError("Error! Cannot call label_node_addition() " + \
                "after finish_labeling() is called.")

        if self.__check_inputs__:
            if len(other_nodes) != self.__subgraph_size__ - 2:
                raise ValueError("Error! Expected `other_nodes` to be" +\
                    " of size %d, was %d." % \
                    (self.__subgraph_size__ - 2, len(other_nodes)))
            if not self.__graph_data__.has_node(node):
                raise ValueError("Error! Need graph_data to have node " +\
                    ("%d in order to label it's addition." % node) + \
                    "(Additions are required to be present in graph in " + \
                    "order to label them.).")
            if not self.__graph_data__.has_edge(edge[0], edge[1]):
                raise ValueError("Error! graph_data not have edge " + \
                    str(edge) + "(Additions are required to be present in " + \
                    "graph in order to label them.).")

        # If directed, highlights go from source (0) to target (1).
        # If undirected, both highlights are 0, unless differentiate endpoints
        #   is set in which case the NEW NODE is given the highlight of 1. Note
        #   that this is different from how edge changes are labeled.
        if self.__graph_data__.is_directed():
            highlights = {edge[0]: 0, edge[1]: 1}
            special_nodes = [edge[0], edge[1]]
        elif self.__diff_ends__:
            non_node = edge[0]
            if non_node == node:
                non_node = edge[1]
            highlights = {non_node: 0, node: 1}
            special_nodes = [non_node, node]
        else:
            highlights = {edge[0]: 0, edge[1]: 0}
            special_nodes = [edge[0], edge[1]]

        if self.__precompute__:
            graph_hash = self.__subgraph_representation__(\
                special_nodes + other_nodes, highlights)
        else:
            (node_order, _) = self.__canonical_node_order__(\
                special_nodes + other_nodes, highlights)
            graph_hash = self.__subgraph_representation__(node_order,highlights)
            if graph_hash not in self.__repr_to_label__[self.__ADD_NODE__]:
                self.__repr_to_label__[self.__ADD_NODE__][graph_hash] = \
                    self.__num_labels__[self.__ADD_NODE__]
                self.__label_to_canonical_repr__[self.__ADD_NODE__][\
                    self.__num_labels__[self.__ADD_NODE__]] = graph_hash
                self.__num_labels__[self.__ADD_NODE__] += 1

        return self.__repr_to_label__[self.__ADD_NODE__][graph_hash]

    # Input:
    #   `node` -- the node in the subgraph that is about to be deleted
    #   `other_nodes` -- a list of the nodes forming the subgraph that are NOT
    #       `node`
    #
    # Output:
    #    (automorphism-invariant) label of the node addition
    #
    # Note:
    #    For computations concerns*, it is expected that the subgraph here is
    #    smaller than the subgraphs for the other labeling functions because the
    #    single node being deleted can have many connections whereas an added
    #    node has only one connection.
    #    * concerns for the application using this class, not this class itself
    def label_node_deletion(self, node, other_nodes):
        if self.__finished_labeling__:
            raise ValueError("Error! Cannot call label_node_deletion() " + \
                "after finish_labeling() is called.")

        if self.__check_inputs__:
            # NOTE: Here, for deletions, the subgraph size is usually 1 smaller
            #   than for additions or edge modifications.
            if len(other_nodes) != \
                    self.__subgraph_size__ - 1 + self.__deletion_modifier__:
                raise ValueError("Error! Expected `other_nodes` to be" +\
                    " of size %d, was %d." % (self.__subgraph_size__ - 1 + \
                    self.__deletion_modifier__, len(other_nodes)))
            if not self.__graph_data__.has_node(node):
                raise ValueError("Error! Need graph_data to have node " +\
                    ("%d in order to label it's deletion." % node))

        # Deleted node given a highlight of 0.
        highlights = {node: 0}
        special_nodes = [node]
        
        if self.__precompute__:
            graph_hash = self.__subgraph_representation__(\
                special_nodes + other_nodes, highlights)
        else:
            (node_order, _) = self.__canonical_node_order__(\
                special_nodes + other_nodes, highlights)
            graph_hash = self.__subgraph_representation__(node_order,highlights)
            if graph_hash not in self.__repr_to_label__[self.__DEL_NODE__]:
                self.__repr_to_label__[self.__DEL_NODE__][graph_hash] = \
                    self.__num_labels__[self.__DEL_NODE__]
                self.__label_to_canonical_repr__[self.__DEL_NODE__][\
                    self.__num_labels__[self.__DEL_NODE__]] = graph_hash
                self.__num_labels__[self.__DEL_NODE__] += 1

        return self.__repr_to_label__[self.__DEL_NODE__][graph_hash]

    # Input:
    #   `label` -- One of the labels the labeler provides.
    #   `change_type` -- an enum from the GraphChange class. One of:
    #       GraphChange.EDGE_ADDITION
    #       GraphChange.EDGE_DELETION
    #       GraphChange.NODE_ADDITION
    #       GraphChange.NODE_DELETION
    #
    # Output:
    #    A "canonical" expression of the subgraph change.
    def get_representative_subgraph_change_from_label(self, label, change_type):
        local_change_type = self.__official_change_type_to_local__(change_type)

        if self.__check_inputs__:
            if label < 0 or self.__num_labels__[local_change_type] <= label:
                raise ValueError("Error! No graph change of type " + \
                    ("%s (locally, %d) " % (change_type, local_change_type)) + \
                    "with label %d." % label)

        return self.__label_to_canonical_repr__[local_change_type][label]

    # Input:
    #   `rep_change` -- A representative change produced by a different subgraph
    #       labeler.
    #   `change_type` -- an enum from the GraphChange class. One of:
    #       GraphChange.EDGE_ADDITION
    #       GraphChange.EDGE_DELETION
    #       GraphChange.NODE_ADDITION
    #       GraphChange.NODE_DELETION
    #
    # Output:
    #   A label. This change has now been stored in the labeler.
    def label_representative_subgraph_change(self, rep_change, change_type):
        if self.__finished_labeling__:
            raise ValueError("Error! Cannot call " + \
                "label_representative_subgraph_change() " + \
                "after finish_labeling() is called.")

        local_change_type = self.__official_change_type_to_local__(change_type)
        if rep_change in self.__repr_to_label__[local_change_type]:
            return self.__repr_to_label__[local_change_type][rep_change]

        next_label = self.__num_labels__[local_change_type]
        self.__num_labels__[local_change_type] += 1
        self.__repr_to_label__[local_change_type][rep_change] = next_label
        self.__label_to_canonical_repr__[local_change_type][next_label] = \
            rep_change
        return next_label

    # Input:
    #   `label` -- One of the labels the labeler provides.
    #   `change_type` -- an enum from the GraphChange class. One of:
    #       GraphChange.EDGE_ADDITION
    #       GraphChange.EDGE_DELETION
    #       GraphChange.NODE_ADDITION
    #       GraphChange.NODE_DELETION
    def num_of_change_cases_for_type(self, change_type):
        local_change_type = self.__official_change_type_to_local__(change_type)
        return self.__num_labels__[local_change_type]

    # Returns all the different subgraph -> subgraph possibilities for adding
    #   an edge.
    def number_of_edge_deletion_cases(self):
        return self.__num_labels__[self.__DEL_EDGE__]

    # Returns all the different subgraph -> subgraph possibilities for adding
    #   an edge.
    def number_of_edge_addition_cases(self):
        return self.__num_labels__[self.__ADD_EDGE__]

    # Assumes a node joins by connecting to a single other node.
    #
    # Since one node has zero neighbors, this is similar to finding the number
    #   of distinct automorphism orbits on graphs one node smaller than those
    #   graphs considered for edge modifications.
    #
    # In an undirected graph with no node or edge properties, this should be
    #   the same number as the number of node deletion cases.
    def number_of_node_addition_cases(self):
        return self.__num_labels__[self.__ADD_NODE__]

    # Assumes a node can be deleted at any time.
    #
    # If `deletion_modifier` is -1 (`deletion_reduced` was True in constructor)
    #   then this looks at one fewer nodes than the normal subgraph_size.
    def number_of_node_deletion_cases(self):
        return self.__num_labels__[self.__DEL_NODE__]

    # Assumes that no more calls to label_...() will be made.
    #
    # Reduces internal storage so that the change labeler can be passed around
    #   in memory more easily.
    def finish_labeling(self):
        self.__finished_labeling__ = True
        self.__repr_to_label__ = {}

    def __official_change_type_to_local__(self, change_type):
        if change_type == GraphChange.EDGE_ADDITION:
            return self.__ADD_EDGE__
        elif change_type == GraphChange.EDGE_DELETION:
            return self.__DEL_EDGE__
        elif change_type == GraphChange.NODE_ADDITION:
            return self.__ADD_NODE__
        elif change_type == GraphChange.NODE_DELETION:
            return self.__DEL_NODE__
        else:
            raise ValueError("Error! `change_type` was not one of the four " + \
                "GraphChange class's change types.")

    # Helper function for the above labeler functions
    def __label_edge_change__(self, edge, other_nodes, change_type):
        # If directed, highlights go from source (0) to target (1).
        # If undirected, both highlights are 0 unless differentiating endpoints,
        #   in which case goes from edge[0] to edge[1].
        if self.__graph_data__.is_directed() or self.__diff_ends__:
            highlights = {edge[0]: 0, edge[1]: 1}
        else:
            highlights = {edge[0]: 0, edge[1]: 0}
        special_nodes = [edge[0], edge[1]]

        if self.__precompute__:
            graph_hash = self.__subgraph_representation__(\
                special_nodes + other_nodes, highlights)
        else:
            (node_order, _) = self.__canonical_node_order__(\
                special_nodes + other_nodes, highlights)
            graph_hash = self.__subgraph_representation__(node_order,highlights)
            if graph_hash not in self.__repr_to_label__[change_type]:
                self.__repr_to_label__[change_type][graph_hash] = \
                    self.__num_labels__[change_type]
                self.__label_to_canonical_repr__[change_type][\
                    self.__num_labels__[change_type]] = graph_hash
                self.__num_labels__[change_type] += 1

        # print(graph_hash)
        # edge_lists = [gh[5] for gh, label in self.__repr_to_label__[change_type].items()]
        # for edge_list in edge_lists:
        #     print("%s ----- %s" % (edge_list, edge_list == graph_hash[5]))
        return self.__repr_to_label__[change_type][graph_hash]

    def __enumerate_and_label_all_possible_inputs__(self):
        # Rather than enumerating all
        #   2^(subgraph_size-choose-2 * (1 + is_directed))
        #   possible naive graph options, builds them up using the
        #   node order canonicalizer, adding one edge (or trait) at a time.
        if self.__graph_data__.is_directed():
            empty_graph = DirectedGraphData()
        else:
            empty_graph = GraphData()

        # To enumerate all possible inputs efficiently, we first generate the
        #   possible graphs do not contain the nodes being added, deleted, or
        #   (dis)connected by a (old)new edge. Thus we iterate over graphs of
        #   up to subgraph_size - 2 nodes.
        #
        # Edge modification examples will add two more nodes: the two nodes
        #   being (dis)connected.
        # Node addition cases will add two more nodes: the new node and the node
        #   that the new node connects to.
        # Node deletion cases will add only one more node: the node being
        #   deleted.

        partial_size = self.__subgraph_size__ - 2
        enumeration_nodes = [i for i in range(0, partial_size)]

        for node in enumeration_nodes:
            empty_graph.add_node(node)

        directed = int(self.__graph_data__.is_directed())

        # Max number of possible edges.
        full_edges = \
            int(((partial_size * (partial_size -1 )) / 2) * (1 + directed))
        # Half of the max possible edges, rounded down.
        half_edges = int(full_edges / 2)
        take_complement_of_half = half_edges * 2 < full_edges

        partial_graph_bank = set()

        # `partial_graphs` functions both as a collection and a queue.
        partial_graphs = [empty_graph]
        next_graph_idx = 0
        partial_graphs_back_half = []

        while next_graph_idx < len(partial_graphs):
            graph = partial_graphs[next_graph_idx]
            next_graph_idx += 1

            # Check to see if the complement of the graph should also be added.
            if graph.num_edges() < half_edges or take_complement_of_half:
                complement = empty_graph.copy()
                for i in range(0, partial_size):
                    for j in range((i + 1) * (1 - directed), partial_size):
                        if not graph.has_edge(i, j):
                            complement.add_edge(i, j)
                partial_graphs_back_half.append(complement)

            # Cycle through possible additions to the graph, checking to see if
            #   they have yet to be found. If so, add them to the collection.
            if graph.num_edges() < half_edges:
                for i in range(0, partial_size):
                    for j in range((i + 1) * (1 - directed), partial_size):
                        if i == j:
                            continue
                        if not graph.has_edge(i, j):
                            copy = graph.copy()
                            copy.add_edge(i, j)
                            canonicalizer = SubgraphChangeLabeler(copy, \
                                subgraph_size=None, precompute=False)
                            (node_order, _) = canonicalizer.\
                                __canonical_node_order__(enumeration_nodes)
                            graph_hash = canonicalizer.\
                                __subgraph_representation__(node_order)
                            if graph_hash not in partial_graph_bank:
                                partial_graph_bank.add(graph_hash)
                                partial_graphs.append(copy)

        for i in range(0, len(partial_graphs_back_half)):
            # Reverse the order of the back half list so that graphs come paired
            #   by complement (first graph complement of last, second complement
            #   of second-to-last, etc.). This way the graphs are also in
            #   order of increasing number of edges.
            partial_graphs.append(partial_graphs_back_half[-1 * (i + 1)])


        # Now that we have acquired all the partial graphs, we add in the nodes
        #   being (primarily) modified and partition the subgraphs according to
        #   which kind(s) of change(s) they represent.
        # We still are not applying traits to the graphs. That happens last.
        edge_change_graphs_wo_traits = []
        node_add_graphs_wo_traits = []
        node_del_graphs_wo_traits = []

        # First we add one new node.
        new_node = partial_size

        possible_edges_first_node = [(new_node, i) for i in range(0, new_node)]
        if directed == 1:
            possible_edges_first_node += \
                [(b, a) for (a, b) in possible_edges_first_node]

        first_node_graph_bank = set()
        first_node_graphs = []
        enumeration_nodes = [i for i in range(0, partial_size + 1)]

        # Highlights are essentially initial labels that distinguish these nodes
        #   from the others in the graph.
        highlights={new_node: 0}

        for partial_graph in partial_graphs:

            edge_combos_first_node = \
                a_utils.get_all_k_tuples(2, len(possible_edges_first_node))

            # Cycle through all the ways the first node could connect to the
            #   partial graphs.
            for edge_combo in edge_combos_first_node:
                copy = partial_graph.copy()
                copy.add_node(new_node)
                for edge_idx in range(0, len(edge_combo)):
                    if edge_combo[edge_idx]:
                        (a, b) = possible_edges_first_node[edge_idx]
                        copy.add_edge(a, b)

                # Check to see if the graph is even new.
                canonicalizer = SubgraphChangeLabeler(copy, subgraph_size=None,\
                    precompute=False)
                (node_order, _) = canonicalizer.__canonical_node_order__(\
                    enumeration_nodes, highlights=highlights)
                graph_hash = canonicalizer.__subgraph_representation__(\
                    node_order, highlights=highlights)
                if graph_hash not in first_node_graph_bank:
                    first_node_graph_bank.add(graph_hash)
                    first_node_graphs.append(copy)

                    # Check to see if copy is a valid node deletion graph by
                    #   checking to see that the graph is connected.
                    if len(g_utils.connected_components(copy)) == 1:
                        node_del_graphs_wo_traits.append(copy)

        del first_node_graph_bank

        # We now commence adding in a second node to the graph.
        new_node += 1
        highlights[new_node] = 0

        # If an ordering is imposed on the nodes, give them different highlights.
        if directed == 1 or self.__diff_ends__:
            highlights[new_node] = 1

        possible_edges_second_node = [(new_node, i) for i in range(0, new_node)]
        if directed == 1:
            possible_edges_second_node += \
                [(b, a) for (a, b) in possible_edges_second_node]

        second_node_graph_bank = set()
        enumeration_nodes = [i for i in range(0, partial_size + 2)]

        for first_node_graph in first_node_graphs:

            edge_combos_second_node = \
                a_utils.get_all_k_tuples(2, len(possible_edges_second_node))

            # Cycle through all the ways the first node could connect to the
            #   partial graphs.
            for edge_combo in edge_combos_second_node:
                copy = first_node_graph.copy()
                copy.add_node(new_node)
                for edge_idx in range(0, len(edge_combo)):
                    if edge_combo[edge_idx]:
                        (a, b) = possible_edges_second_node[edge_idx]
                        copy.add_edge(a, b)

                # Check to see if the graph is even new.
                canonicalizer = SubgraphChangeLabeler(copy, subgraph_size=None,\
                    precompute=False)
                (node_order, _) = canonicalizer.__canonical_node_order__(\
                    enumeration_nodes, highlights=highlights)
                graph_hash = canonicalizer.__subgraph_representation__(\
                    node_order, highlights=highlights)
                if graph_hash not in second_node_graph_bank:
                    second_node_graph_bank.add(graph_hash)

                    connected_components = g_utils.connected_components(copy)
                    num_connected_components = len(connected_components)

                    # Check to see if copy is a valid node addition graph by
                    #   checking to see that the graph is connected and that
                    #   one of the highlighted nodes connects only to the other
                    #   highlighted node.
                    #
                    # If the graph is directed, ensure that the edge points from
                    #   (new_node - 1) to new_node so that the highlights follow
                    #   the source->target convention.
                    if num_connected_components == 1:

                        if directed == 1 and \
                           ((copy.in_neighbors(new_node)==set([new_node-1]) and\
                                len(copy.out_neighbors(new_node)) == 0) or \
                           (copy.out_neighbors(new_node-1)==set([new_node]) and\
                                len(copy.in_neighbors(new_node - 1)) == 0)):
                            node_add_graphs_wo_traits.append(copy)

                        elif directed == 0 and self.__diff_ends__ and \
                                copy.neighbors(new_node) == set([new_node-1]):
                            node_add_graphs_wo_traits.append(copy)

                        elif directed == 0 and (not self.__diff_ends__) and \
                               (copy.neighbors(new_node) == set([new_node-1]) or
                               copy.neighbors(new_node - 1) == set([new_node])):
                            node_add_graphs_wo_traits.append(copy)

                    # Check to see if copy is a valid edge modification graph.
                    #
                    # Note that if the graph is directed, we enforce that the
                    #   direction be from (new_node - 1) to new_node, but if it
                    #   is undirected, the has_edge() function looks for an edge
                    #   in either direction.
                    if num_connected_components == 1 and \
                            (copy.has_edge(new_node - 1, new_node)):
                        edge_change_graphs_wo_traits.append(copy.copy())

        del second_node_graph_bank

        print("Number of Node Deletion Cases (W/O Traits): \t%d" % len(node_del_graphs_wo_traits))
        print("Number of Node Addition Cases (W/O Traits): \t%d" % len(node_add_graphs_wo_traits))
        print("Number of Edge Modification Cases (W/O Traits): \t%d" % len(edge_change_graphs_wo_traits))

        ###### Now for the node and edge traits. #######

        # Note that `highlights` acquires its value above.

        graph_lists_wo_traits = \
            [node_del_graphs_wo_traits, node_add_graphs_wo_traits, \
             edge_change_graphs_wo_traits]
        graph_lists_with_traits = [[], [], []]

        if len(self.__node_traits__) == 0 and len(self.__edge_traits__) == 0:
            # No traits to handle. Moving on.
            graph_lists_with_traits = graph_lists_wo_traits

        # Apply all possible trait combos.
        else:
            for change_type_idx in range(0, 3):
                # Reset the graph hash bank for each type of change.
                graph_trait_bank = set()

                graph_list_wo_traits = graph_lists_wo_traits[change_type_idx]
                graph_list_with_traits = graph_lists_with_traits[change_type_idx]

                for graph in graph_list_wo_traits:

                    nodes = [i for i in range(0, graph.num_nodes())]
                    edges = []
                    for i in range(0, self.__subgraph_size__):
                        for j in range((i + 1) * (1 - directed), \
                                self.__subgraph_size__):
                            if graph.has_edge(i, j):
                                edges.append((i, j))

                    entity_sets = [nodes, edges]
                    trait_sets = [self.__node_traits__, self.__edge_traits__]
                    trait_values_sets = [[], []]
                    for trait_set_idx in range(0, 2):
                        traits = trait_sets[trait_set_idx]
                        entities = entity_sets[trait_set_idx]
                        trait_values = trait_values_sets[trait_set_idx]
                        n_ent = len(entities)

                        for trait in traits:
                            graph.add_trait(trait[1])

                            if trait[0] == SubgraphChangeLabeler.RANK_TRAIT:
                                # It's a rank trait.
                                # If guaranteed unique, only need permutations
                                #   for rankings.
                                if trait[2]:
                                    trait_values.append(a_utils.\
                                        get_all_k_permutations(n_ent, n_ent))
                                # Otherwise, allow for ties.
                                else:
                                    trait_values.append(\
                                        a_utils.get_all_n_rankings(n_ent))
                            else:
                                # It's a class trait.
                                options = trait[2]
                                combos = a_utils.get_all_k_tuples(\
                                        len(options), n_ent)
                                trait_values.append(\
                                    [[options[idx] for idx in combo] \
                                        for combo in combos])

                    trait_names = [t[1] for t in self.__node_traits__] + \
                        [t[1] for t in self.__edge_traits__]
                    entities_by_trait = [nodes for t in self.__node_traits__] +\
                        [edges for t in self.__edge_traits__]
                    values_by_trait = trait_values_sets[0] +trait_values_sets[1]

                    trait_combo_counter = \
                        [0 for i in range(0, len(entities_by_trait))]
                    trait_combo_counter[-1] = -1

                    digit_limits_inclusive = \
                        [len(values) - 1 for values in values_by_trait]

                    while a_utils.increment_counter(trait_combo_counter, \
                            digit_limits_inclusive):

                        copy = graph.copy()
                        for trait_idx in range(0, len(trait_names)):
                            entities = entities_by_trait[trait_idx]
                            values_selection = trait_combo_counter[trait_idx]
                            values = values_by_trait[trait_idx][values_selection]
                            for ent_idx in range(0, len(entities)):
                                ent = entities[ent_idx]
                                copy[trait_names[trait_idx]][ent] = \
                                    values[ent_idx]

                        # Trait values have been assigned to copy. Now check to
                        #   see if this assignment is isomorphically unique.
                        canonicalizer = SubgraphChangeLabeler(copy, \
                            node_traits=self.__node_traits__, \
                            edge_traits=self.__edge_traits__, \
                            subgraph_size=None, precompute=False)
                        (node_order, _) = canonicalizer.\
                            __canonical_node_order__(nodes, highlights)
                        graph_hash = canonicalizer.\
                            __subgraph_representation__(node_order, highlights)
                        if graph_hash not in graph_trait_bank:
                            graph_trait_bank.add(graph_hash)
                            graph_list_with_traits.append(copy)

            del graph_trait_bank

        print("Number of Node Deletion Cases (With Traits): \t%d" % len(graph_lists_with_traits[0]))
        print("Number of Node Addition Cases (With Traits): \t%d" % len(graph_lists_with_traits[1]))
        print("Number of Edge Modification Cases (With Traits): \t%d" % len(graph_lists_with_traits[2]))

        ####### Lastly, store possible inputs. #######

        self.__DEL_NODE__ = 0
        self.__ADD_NODE__ = 1
        self.__DEL_EDGE__ = 2
        self.__ADD_EDGE__ = 3
        self.__num_labels__ = {0: 0, 1: 0, 2: 0}
        # Since we're performing a full enumeration and edge additions have all
        #   the same cases as edge deletions, the dicts can be duplicated.
        self.__repr_to_label__ = {0: dict(), 1: dict(), 2: dict()}
        self.__repr_to_label__[3] = self.__repr_to_label__[2]
        self.__label_to_canonical_repr__ = {0: dict(), 1: dict(), 2: dict()}
        self.__label_to_canonical_repr__[3] =self.__label_to_canonical_repr__[2]

        for change_type in range(0, 3):
            num_nodes = graph_lists_with_traits[change_type][0].num_nodes()
            nodes = [i for i in range(0, num_nodes)]

            num_highlight_nodes = 1 + int(num_nodes == self.__subgraph_size__)
            num_other_nodes = num_nodes - num_highlight_nodes

            other_nodes_orders = \
                a_utils.get_all_k_permutations(num_other_nodes, num_other_nodes)
            other_nodes_orders = [[n + num_highlight_nodes for n in order] for \
                order in other_nodes_orders]

            if directed == 1 or self.__diff_ends__:
                highlight_nodes_orders = \
                    [tuple([i for i in range(0, num_highlight_nodes)])]
                highlights = {i: i for i in range(0, num_highlight_nodes)}
            else:
                highlight_nodes_orders = a_utils.get_all_k_permutations(\
                    num_highlight_nodes, num_highlight_nodes)
                highlights = {i: 0 for i in range(0, num_highlight_nodes)}

            for graph in graph_lists_with_traits[change_type]:

                label = self.__num_labels__[change_type]
                self.__num_labels__[change_type] += 1

                # First create the canonical label.
                canonicalizer = SubgraphChangeLabeler(graph, \
                    node_traits=self.__node_traits__, \
                    edge_traits=self.__edge_traits__, \
                    subgraph_size=None, precompute=False)
                (order, _) = \
                    canonicalizer.__canonical_node_order__(nodes, highlights)
                graph_hash = \
                    canonicalizer.__subgraph_representation__(order, highlights)
                self.__label_to_canonical_repr__[change_type][label] =graph_hash

                # Then enumerate possible input orders.
                for highlight_order in highlight_nodes_orders:
                    highlight_order_list = list(highlight_order)
                    for other_order in other_nodes_orders:
                        node_order = highlight_order_list + other_order
                        graph_hash = canonicalizer.__subgraph_representation__(\
                            node_order, highlights)
                        self.__repr_to_label__[change_type][graph_hash] = label

        self.__num_labels__[3] = self.__num_labels__[2]

        change_type_names = ["node deletion", "node addition", "edge deletion", "edge addition"]
        for change_type in range(0, 4):
            print("Number %s cases with possible input orders: %d" % (change_type_names[change_type], len(self.__repr_to_label__[change_type])))


    # Assumes of `nodes` that if order matters, the nodes are in that order.
    #
    # Returns a hashable representation of the subgraph that includes any
    #   highlights from the nodes dict and all the node and edge trait
    #   information. Intended to be (somewhat) human readable.
    #
    # `highlighted_nodes_dict` is, in practice, just used to highlight nodes
    #   which are involved in a change. Conceptually though, this code should
    #   work for any initial highlights.
    def __subgraph_representation__(self, nodes, highlights={}):
        # If no initial order is provided, create an arbitrary consistent order.
        if type(nodes) is not list:
            nodes = list(nodes)

        node_highlights =\
            tuple([highlights[n] if n in highlights else None for n in nodes])

        node_traits = []
        for trait in self.__node_traits__:
            trait_name = trait[1]
            trait_values = \
                self.__get_trait_values_from_graph_data__(nodes, trait)
            node_traits.append("%s:" % trait_name)
            node_traits.append(tuple([trait_values[n] for n in nodes]))
        node_traits = tuple(node_traits)

        # Now for the edge information...
        (edges, local_idx_edges) = self.__enumerate_edges__(nodes)

        edge_list = tuple(local_idx_edges)

        edge_traits = []
        for trait in self.__edge_traits__:
            trait_name = trait[1]
            trait_values = \
                self.__get_trait_values_from_graph_data__(edges, trait)
            edge_traits.append("%s:" % trait_name)
            edge_traits.append(tuple([trait_values[e] for e in edges]))
        edge_traits = tuple(edge_traits)

        return ("Node Highlights:", node_highlights, "Node Traits:", \
            node_traits, "Edge List:", edge_list, "Edge Traits:", edge_traits)

    # Utilizes Weisfeiler Lehman to determine a canonical node order for _small_
    #   (7-node or fewer?) subgraphs. Can handle node and edge traits as well.
    #
    # "Canonical" here means automorphism-invariant.
    #
    # Designed to work for directed and undirected graphs.
    def __canonical_node_order__(self, nodes, highlights={}):
        if type(nodes) is not list:
            nodes = list(nodes)

        # Highlighted nodes come first.
        max_label = max([-1] + [l for n, l in highlights.items()]) + 1
        node_labels = {n: [highlights[n]] if n in \
            highlights else [max_label] for n in nodes}

        # Add node traits to node labels in order of trait.
        for trait in self.__node_traits__:
            trait_values = \
                self.__get_trait_values_from_graph_data__(nodes, trait)
            for node in nodes:
                node_labels[node].append(trait_values[node])

        # Give nodes numeric labels based on the larger labels' sorted order.
        node_labels_list = [(l, n) for n, l in node_labels.items()]
        node_labels_list.sort()
        numeric_label = -1
        prev_label = None
        node_partitioning = []
        for (label, node) in node_labels_list:
            if label != prev_label:
                numeric_label += 1
                prev_label = label
                node_partitioning.append([])
            node_labels[node] = numeric_label
            node_partitioning[-1].append(node)

        # Now acquire initial labels for edges as well.
        (edges, _) = self.__enumerate_edges__(nodes)
        # If undirected, create duplicates for direction reversal.
        if not self.__graph_data__.is_directed():
            edges = edges + [(b, a) for (a, b) in edges]

        # Acquire labels based on listing traits in order.
        edge_labels = {edge: [] for edge in edges}
        for trait in self.__edge_traits__:
            trait_values = \
                self.__get_trait_values_from_graph_data__(edges, trait)
            for edge in edges:
                edge_labels[edge].append(trait_values[edge])

        # Give edges numeric labels based on larger labels' sorted order.
        edge_labels_list = [(l, e) for e, l in edge_labels.items()]
        edge_labels_list.sort()
        numeric_label = -1
        prev_label = None
        for (label, edge) in edge_labels_list:
            if label != prev_label:
                numeric_label += 1
                prev_label = label
            edge_labels[edge] = numeric_label

        found_automorphism_orbits = False
        orbits = []

        # Having aquired all labels, now repeatedly perform Weisfeiler Lehman to
        #   create canonical order for nodes.
        while len(node_partitioning) < len(nodes):
            old_num_cells = len(node_partitioning)
            for cell_idx in range(0, old_num_cells):
                cell = node_partitioning[cell_idx]

                if len(cell) == 1:
                    continue  # Cell already individualized.

                # Label a node by a (sorted) list of connections to neighbors,
                #   where each connection is labeled both by the edge label(s)
                #   and the neighbor's label.
                cell_label_list = [(sorted([\
                     (edge_labels[(n, n2)] if (n, n2) in edges else -1,\
                      edge_labels[(n2, n)] if (n2, n) in edges else -1,\
                      node_labels[n2])\
                    for n2 in nodes]), n) for n in cell]
                cell_label_list.sort()

                if cell_label_list[0][0] == cell_label_list[-1][0]:
                    continue  # No changes occurred.

                # Split the cell!
                prev_label = None
                new_cells = []
                for (label, node) in cell_label_list:
                    if label != prev_label:
                        new_cells.append([])
                        prev_label = label
                    new_cells[-1].append(node)
                node_partitioning = node_partitioning[0:cell_idx] + new_cells + \
                    node_partitioning[cell_idx + 1:]

                break

            # If we need to arbitrarily individualize... do so by node label.
            if len(node_partitioning) == old_num_cells:
                if not found_automorphism_orbits:
                    found_automorphism_orbits = True
                    orbits = [list(l) for l in node_partitioning]

                partition_idx = 0
                while len(node_partitioning[partition_idx]) == 1:
                    partition_idx += 1
                node_partitioning[partition_idx].sort()
                node_partitioning = node_partitioning[0:partition_idx] + \
                    [[node_partitioning[partition_idx][0]]] + \
                    [node_partitioning[partition_idx][1:]] + \
                    node_partitioning[partition_idx + 1:]

            # Relabel nodes.
            for cell_idx in range(0, len(node_partitioning)):
                cell = node_partitioning[cell_idx]
                for node in cell:
                    node_labels[node] = cell_idx

        if not found_automorphism_orbits:
            orbits = [list(l) for l in node_partitioning]

        return ([cell[0] for cell in node_partitioning], orbits)

    # `elements` is a list of nodes or edges.
    # `trait` is a trait for which those nodes or edges have values -- should
    #   be made by the GraphChangeModeler.RankTrait() or .ClassTrait() methods.
    #
    # Returns a dict mapping the elements to the values.
    def __get_trait_values_from_graph_data__(self, elements, trait):
        trait_type = trait[0]
        trait_name = trait[1]

        # If it's a class trait, we can just return the class values.
        if trait_type == SubgraphChangeLabeler.CLASS_TRAIT:
            return {e: self.__graph_data__[trait_name][e] for e in elements}

        # Otherwise, it's a rank trait.
        trait_element_pairings = \
            [(self.__graph_data__[trait_name][elt], elt) for elt in elements]
        trait_element_pairings.sort()

        # Associate elements with the rank of their trait_name value.
        # Handle ties by giving those elements the same rank.
        (_, _, guaranteed_unique) = trait
        next_rank = 0
        new_pairings = {trait_element_pairings[0][1]: next_rank}
        past_value = trait_element_pairings[0][0]

        # Used to allow (a, b) and (b, a) to have same value even if
        #   guaranteed_unique is set to True in an undirected graph.
        reverse_edge_exception = \
            (type(elements[0]) is tuple) and not self.__graph_data__.is_directed()
        if reverse_edge_exception:
            past_elt_reverse = trait_element_pairings[0][1]
            past_elt_reverse = (past_elt_reverse[1], past_elt_reverse[0])

        for i in range(1, len(trait_element_pairings)):
            (new_value, new_elt) = trait_element_pairings[i]
            if new_value == past_value:
                if guaranteed_unique and not \
                    (reverse_edge_exception and past_elt_reverse == new_elt):
                        raise ValueError("Error! Expected trait " + \
                            "%s to be unique, but elements %s and %s share " % \
                                (trait_name, trait_element_pairings[i - 1][1], \
                                trait_element_pairings[i][1]) + \
                            "value %s" % (new_value))
            if new_value != past_value:
                past_value = new_value
                next_rank += 1  # Rank increases with every new value.
            new_pairings[new_elt] = next_rank
            if reverse_edge_exception:
                past_elt_reverse = (new_elt[1], new_elt[0])
        return new_pairings

    def __enumerate_edges__(self, nodes):
        edges = []
        local_idx_edges = []
        # These loops are O(len(nodes)^2), but len(nodes) is expected to be
        #   very small (likely 6 or smaller). Thus, this should actually be
        #   faster than looking at nodes' neighbor sets and filtering, which
        #   would be at least O(len(nodes) * average_degree).
        if self.__graph_data__.is_directed():
            for i in range(0, len(nodes)):
                for j in range(0, len(nodes)):
                    if i == j:
                        continue
                    if self.__graph_data__.has_edge(nodes[i], nodes[j]):
                        edges.append((nodes[i], nodes[j]))
                        local_idx_edges.append((i, j))
        else:
            for i in range(0, len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.__graph_data__.has_edge(nodes[i], nodes[j]):
                        edges.append((nodes[i], nodes[j]))
                        local_idx_edges.append((i, j))
        return (edges, local_idx_edges)

if __name__ == "__main__":
    from graph_change_modeler import GraphChangeModeler

    GD = GraphData()
    # GD = DirectedGraphData()
    # node_traits = [GraphChangeModeler.ClassTrait(X, ["A", "B", "C", "D"]) for X in ["t0", "t1"]]
    # node_traits = [GraphChangeModeler.RankTrait("Pagerank", guaranteed_unique=False)]
    # edge_traits = [GraphChangeModeler.RankTrait("Edgerank", guaranteed_unique=False)]
    node_traits = [GraphChangeModeler.ClassTrait("Change", possible_class_values=["Added", "None", "Deleted"])] #, GraphChangeModeler.RankTrait("Rank", guaranteed_unique=False)]
    edge_traits = [GraphChangeModeler.ClassTrait("Change", possible_class_values=["Added", "None", "Deleted"])]
    # node_traits = []
    # edge_traits = []
    SCL = SubgraphChangeLabeler(GD, 4, node_traits=node_traits, edge_traits=edge_traits, \
        deletion_reduced=True, differentiate_endpoints=True)
