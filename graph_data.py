import networkx as nx
from sample_set import SampleSet

"""
This file contains two main classes:
 *  GraphData
 *  DirectedGraphData

They are designed to be compatible with both versions of Python and both
versions of NetworkX.

Both provide the following interface:

 *  FromNetworkX(nx_graph)
        Class method. Returns a graph instantiated by copying data from a
        NetworkX graph. Automatically transfers NetworkX weights and attributes
 *  add_node(node)
 *  delete_node(node)
 *  add_edge(source, target)
        Order does not matter unless using the _directed_ wrapper.
 *  delete_edge(source, target)
        Order does not matter unless using the _directed_ wrapper.
 *  has_node(node)
 *  has_edge(source, target)
        Order does not matter unless using the _directed_ wrapper.
 *  random_node()
        Randomly returns one of the nodes. O(1)
 *  random_edge()
        Randomly returns one of the edges. O(1)
 *  add_trait(name)
        Allows for the addition of an edge or node trait to the graph.
        Traits can be accessed via brackets ( [] ). For example:
        G = DirectedGraphData()
        G.add_node(1)
        G.add_node(2)
        G.add_edge(1, 2)
        G.add_trait("pagerank")
        G["pagerank"][1] = 0.15
        G["pagerank"][2] = 1.85
        G.add_trait("weight")
        G["weight"][(1, 2)] = 1.0

        Note that, for undirected graphs, traits on edges can be accessed with
        either ordering of the nodes. So, if G was just of the GraphData
        class, not DirectedGraphData then one could assert the following:
        assert G["weight"][(2, 1)] == 1.0

 *  nodes()
        Returns a set of all nodes in the graph.

        Note that, for efficiency, this returns an _editable_ set. For safety,
        consider copying. For example:
        nodes = set(G.nodes())

 *  edges()
        Returns a set of all edges in the graph.

        Note that, for efficiency, this returns an _editable_ set. For safety,
        consider copying. For example:
        edges = set(G.edges())

 *  num_edges()
 *  num_nodes()
 *  is_directed()
 *  neighbors(node)
        Note that, for efficiency, this returns an _editable_ set. For safety,
        consider copying. For example:
        neighbors_x = set(G.neighbors(x))
 *  copy()
        Returns a copy of the graph.

        IMPORTANT! Assumes that the trait values are immutable (ints, floats,
        strings, etc.).

 *  set_savepoint(name=None)
        Signals that the graph_data should begin logging all changes.

        IMPORTANT! Cannot set a savepoint if one is already set. `name` is just
        for debugging purposes to help diagnose when this condition is violated.

 *  restore_to_savepoint()
        Undoes all changes after the set savepoint. Note that 'changes' which
        did not actually change the graph are not undone (e.g. adding an edge
        which already existed in the graph.)

        IMPORTANT! Does not clear the savepoint.

 *  clear_savepoint()
        Signals that the graph_data may delete its changelog and stop logging.

In addition, the DirectedGraphData class provides some additional methods:

 *  in_neighbors(node)
        Returns a set of all nodes that point to `node`.

        Note that, for efficiency, this returns an _editable_ set. For safety,
        consider copying. For example:
        in_neighbors_x = set(G.in_neighbors(x))
 *  out_neighbors(node)
        Returns a set of all nodes that `node` points to.

        Note that, for efficiency, this returns an _editable_ set. For safety,
        consider copying. For example:
        out_neighbors_x = set(G.out_neighbors(x))
"""

class GraphData:

    def __init__(self):
        self.__nodes__ = SampleSet()
        self.__edges__ = SampleSet()
        self.__traits__ = {}
        self.__savepoint_set__ = False
        self.__savepoint_name__ = None
        self.__undo_log__ = []

        self.__neighbor_sets__ = {}
        # UndirectedDict is defined below in this file.
        self.__trait_dict_type__ = UndirectedDict

    __ADD_TRAIT__ = 0
    __ADD_NODE__ = 1
    __ADD_EDGE__ = 2
    __DEL_NODE__ = 3
    __DEL_EDGE__ = 4

    @classmethod
    def FromNetworkX(cls, nx_graph):
        v2 = nx.__version__ >= '2.0'

        G = cls()
        for node in nx_graph.nodes():
            G.add_node(node)
            if v2:
                d = nx_graph.nodes[node]
            else:
                d = nx_graph.node[node]
            for attribute, value in d.items():
                if attribute not in G.__traits__:
                    G.add_trait(attribute)
                G[attribute][node] = value

        for (a, b) in nx_graph.edges():
            G.add_edge(a, b)
            if v2:
                d = nx_graph.edges[(a, b)]
            else:
                d = nx_graph.edge[(a, b)]
            for attribute, value in d.items():
                if attribute not in G.__traits__:
                    G.add_trait(attribute)
                G[attribute][(a, b)] = value

        return G

    def add_trait(self, name):
        if name not in self.__traits__:
            self.__traits__[name] = \
                SavepointDictWrapper(self.__trait_dict_type__())
            if self.__savepoint_set__:
                self.__undo_log__.append((GraphData.__ADD_TRAIT__, name))

    def add_node(self, node):
        if node in self.__nodes__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__ADD_NODE__, node))

        self.__nodes__.add(node)
        self.__neighbor_sets__[node] = set()

    def delete_node(self, node):
        if node not in self.__nodes__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__DEL_NODE__, node, \
                [(node, n) for n in self.__neighbor_sets__[node]]))

        for neighbor in self.__neighbor_sets__[node]:
            self.__neighbor_sets__[neighbor].remove(node)
            self.__edges__.remove((min(node, neighbor), max(node, neighbor)))
        del self.__neighbor_sets__[node]
        self.__nodes__.remove(node)
        for trait, trait_dict in self.__traits__.items():
            if node in trait_dict:
                del trait_dict[node]

    def add_edge(self, source, target):
        source_ = min(source, target)
        target_ = max(source, target)

        if (source_, target_) in self.__edges__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__ADD_EDGE__, (source_, target_)))

        self.__edges__.add((source_, target_))
        self.__neighbor_sets__[source].add(target)
        self.__neighbor_sets__[target].add(source)

    def delete_edge(self, source, target):
        self.__neighbor_sets__[source].remove(target)
        self.__neighbor_sets__[target].remove(source)
        source_ = min(source, target)
        target_ = max(source, target)

        edge = (source_, target_)
        if edge not in self.__edges__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__DEL_EDGE__, edge))

        self.__edges__.remove(edge)
        for trait, trait_dict in self.__traits__.items():
            if edge in trait_dict:
                del trait_dict[edge]

    def nodes(self):
        return self.__nodes__

    def edges(self):
        return self.__edges__

    def num_nodes(self):
        return len(self.__nodes__)

    def num_edges(self):
        return len(self.__edges__)

    def is_directed(self):
        return False

    def has_node(self, node):
        return node in self.__nodes__

    def has_edge(self, source, target):
        return (min(source, target), max(source, target)) in self.__edges__

    def random_node(self):
        return self.__nodes__.randomly_sample()

    def random_edge(self):
        return self.__edges__.randomly_sample()

    # Caution: Returns editable copy!
    def neighbors(self, node):
        return self.__neighbor_sets__[node]

    def __getitem__(self, key):
        if key not in self.__traits__:
            raise ValueError("Error! Trait %s not found in graph. " % key + \
                "Use add_trait(%s) first." % key)
        return self.__traits__[key]

    def __setitem__(self, key, value):
        raise ValueError("Error! Traits must be set via add_trait().")

    def copy(self):
        if self.is_directed():
            c = DirectedGraphData()
        else:
            c = GraphData()
        for node in self.__nodes__:
            c.add_node(node)
        for (a, b) in self.__edges__:
            c.add_edge(a, b)
        for trait_name, trait_dict in self.__traits__.items():
            c.add_trait(trait_name)
            for element, trait_value in trait_dict.items():
                c[trait_name][element] = trait_value
        return c

    # `name` is for debugging purposes only.
    def set_savepoint(self, name=None):
        if self.__savepoint_set__:
            if self.__savepoint_name__ is not None:
                old_name_str = " (name of previous savepoint: %s)" % \
                    self.__savepoint_name__
            else:
                old_name_str = ""
            raise ValueError("Error! Setting a graph_data savepoint when " + \
                "one is already set!" + old_name_str)
        self.__savepoint_name__ = name
        self.__savepoint_set__ = True
        self.__undo_log__ = []
        for _, trait_dict in self.__traits__.items():
            trait_dict.set_savepoint()

    def restore_to_savepoint(self):
        # Temporarily prevent these 'changes' from being put in the log.
        self.__savepoint_set__ = False
        for _, trait_dict in self.__traits__.items():
            trait_dict.restore_to_savepoint()
            # Don't bother recording the 'changes' made below.
            trait_dict.clear_savepoint()

        for i in range(0, len(self.__undo_log__)):
            undo_data = self.__undo_log__[(len(self.__undo_log__) - 1) - i]
            undo_type = undo_data[0]
            if undo_type == GraphData.__ADD_TRAIT__:
                del self.__traits__[undo_data[1]]
            elif undo_type == GraphData.__ADD_NODE__:
                self.delete_node(undo_data[1])
            elif undo_type == GraphData.__ADD_EDGE__:
                self.delete_edge(undo_data[1][0], undo_data[1][1])
            elif undo_type == GraphData.__DEL_NODE__:
                self.add_node(undo_data[1])
                for edge in undo_data[2]:
                    self.add_edge(edge[0], edge[1])
            else:  # GraphData.__DEL_EDGE__
                self.add_edge(undo_data[1][0], undo_data[1][1])

        # Restore the savepoint_set status.
        self.__savepoint_set__ = True
        for _, trait_dict in self.__traits__.items():
            trait_dict.set_savepoint()

    def clear_savepoint(self):
        self.__savepoint_set__ = False
        self.__undo_log__ = []
        self.__savepoint_name__ = None
        for _, trait_dict in self.__traits__.items():
            trait_dict.clear_savepoint()

class DirectedGraphData(GraphData):

    def __init__(self):
        self.__nodes__ = SampleSet()
        self.__edges__ = SampleSet()
        self.__traits__ = {}
        self.__neighbor_sets__ = {}

        self.__savepoint_set__ = False
        self.__savepoint_name__ = None
        self.__undo_log__ = []
        #
        self.__in_neighbor_sets__ = {}
        self.__out_neighbor_sets__ = {}
        self.__trait_dict_type__ = dict

    def add_node(self, node):
        if node in self.__nodes__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__ADD_NODE__, node))

        self.__nodes__.add(node)
        self.__neighbor_sets__[node] = set()
        #
        self.__in_neighbor_sets__[node] = set()
        self.__out_neighbor_sets__[node] = set()

    def delete_node(self, node):
        if node not in self.__nodes__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__DEL_NODE__, node, \
                [(node, n) for n in self.__out_neighbor_sets__[node]] + \
                [(n, node) for n in self.__in_neighbor_sets__[node]]))

        for neighbor in self.__neighbor_sets__[node]:
            self.__neighbor_sets__[neighbor].remove(node)
        del self.__neighbor_sets__[node]
        self.__nodes__.remove(node)
        for trait, trait_dict in self.__traits__.items():
            if node in trait_dict:
                del trait_dict[node]
        #
        for neighbor in self.__in_neighbor_sets__[node]:
            self.__out_neighbor_sets__[neighbor].remove(node)
            self.__edges__.remove((neighbor, node))
        for neighbor in self.__out_neighbor_sets__[node]:
            self.__in_neighbor_sets__[neighbor].remove(node)
            self.__edges__.remove((node, neighbor))

        del self.__in_neighbor_sets__[node]
        del self.__out_neighbor_sets__[node]

    def add_edge(self, source, target):
        if (source, target) in self.__edges__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__ADD_EDGE__, (source, target)))

        self.__edges__.add((source, target))
        self.__neighbor_sets__[source].add(target)
        self.__neighbor_sets__[target].add(source)
        #
        self.__out_neighbor_sets__[source].add(target)
        self.__in_neighbor_sets__[target].add(source)

    def delete_edge(self, source, target):
        if (source, target) not in self.__edges__:
            return

        if self.__savepoint_set__:
            self.__undo_log__.append((GraphData.__DEL_EDGE__, (source, target)))

        if (target, source) not in self.__edges__:
            self.__neighbor_sets__[source].remove(target)
            self.__neighbor_sets__[target].remove(source)
        self.__in_neighbor_sets__[target].remove(source)
        self.__out_neighbor_sets__[source].remove(target)
        edge = (source, target)
        self.__edges__.remove(edge)
        for trait, trait_dict in self.__traits__.items():
            if edge in trait_dict:
                del trait_dict[edge]

    def is_directed(self):
        return True

    def has_edge(self, source, target):
        return (source, target) in self.__edges__

    # Caution: Returns editable copy!
    def out_neighbors(self, node):
        return self.__out_neighbor_sets__[node]

    # Caution: Returns editable copy!
    def in_neighbors(self, node):
        return self.__in_neighbor_sets__[node]

# Used as a wrapper for any dict-esque object, allowing the setting of a single
#   savepoint for that object.
class SavepointDictWrapper:

    def __init__(self, base_dict):
        self.__d__ = base_dict
        self.__undo_records__ = type(base_dict)()
        self.__savepoint_set__ = False

    def __getitem__(self, key):
        return self.__d__[key]

    def __setitem__(self, key, value):
        if self.__savepoint_set__ and key not in self.__undo_records__:
            if key in self.__d__:
                self.__undo_records__[key] = (True, self.__d__[key])
            else:
                self.__undo_records__[key] = (False,)
        self.__d__[key] = value

    def __delitem__(self, key):
        if self.__savepoint_set__ and key not in self.__undo_records__:
            self.__undo_records__[key] = (True, self.__d__[key])
        del self.__d__[key]

    def __contains__(self, key):
        return key in self.__d__

    # Use with care.
    def items(self):
        return self.__d__.items()

    def set_savepoint(self):
        self.__savepoint_set__ = True
        self.__undo_records__ = type(self.__d__)()

    def restore_to_savepoint(self):
        for key, undo_record in self.__undo_records__.items():
            if undo_record[0]:
                self.__d__[key] = undo_record[1]
            else:
                del self.__d__[key]

    def clear_savepoint(self):
        self.__savepoint_set__ = False
        self.__undo_records__ = type(self.__d__)()

# Used as a dictionary that, when the key is a tuple, automatically orders the
# first two elements (effectively allows for undirected edges to be accessed
# with either ordering).
# Specifically, used for traits.
class UndirectedDict(object):

    def __init__(self):
        self.__d__ = {}

    def __real_key__(self, key):
        if type(key) is tuple:
            return (min(key[0], key[1]), max(key[0], key[1]))
        return key

    def __getitem__(self, key):
        return self.__d__[self.__real_key__(key)]

    def __setitem__(self, key, value):
        self.__d__[self.__real_key__(key)] = value

    def __delitem__(self, key):
        del self.__d__[self.__real_key__(key)]

    def __contains__(self, key):
        return self.__real_key__(key) in self.__d__

    # Use with care.
    def items(self):
        return self.__d__.items()

if __name__ == "__main__":

    dict_1 = {1: "hello", 3: "gelatin"}
    sdw = SavepointDictWrapper(dict_1)
    sdw.set_savepoint()
    sdw[3] = "sentence"
    sdw[2] = "world"
    sdw.set_savepoint()
    del sdw[1]
    sdw[2] = "unhappy"
    sdw[1] = "overwritten"
    print(sdw.items())
    sdw.restore_to_savepoint()
    print(sdw.items())

    dict_2 = UndirectedDict()
    usdw = SavepointDictWrapper(dict_2)
    usdw[(1, 0)] = "(0, 1)A"
    usdw[(0, 2)] = "(0, 2)A"
    usdw.set_savepoint()
    usdw[(0, 1)] = "(0, 1)B"
    usdw[(1, 0)] = "(0, 1)C"
    del usdw[(2, 0)]
    print(usdw.items())
    usdw.restore_to_savepoint()
    print(usdw.items())
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=0.5)
    G.add_edge(2, 3, weight=0.1)
    G.add_edge(2, 1, weight=0.75)
    T1 = GraphData.FromNetworkX(G)
    T1.set_savepoint()
    print("\n------------------------------------------------")
    print("T1 at savepoint:")
    print("Edges: %s" % T1.__edges__)
    print("Weights: %s" % T1['weight'].items())
    print("Undirected Check: %s" % (T1['weight'][(1, 2)] == T1['weight'][(2, 1)]))
    T1.add_edge(1, 3)
    T1['weight'][(2, 3)] = 0.2
    T1['weight'][(1, 3)] = 0.5
    T1.delete_node(3)
    T1.add_node(4)
    T1.add_edge(1, 4)
    T1.delete_edge(1, 2)
    T1.add_trait("New Trait")
    T1['weight'][(1, 4)] = 0.25
    print("T1 after changes:")
    print("Edges: %s" % T1.__edges__)
    print("Weights: %s" % T1['weight'].items())
    T1.restore_to_savepoint()
    print("T1 restored to savepoint:")
    print("Edges: %s" % T1.__edges__)
    print("Weights: %s" % T1['weight'].items())
    print("------------------------------------------------\n")

    T1.clear_savepoint() 
    print("The Critical Test For Undirected:")
    edges_before = set(T1.edges())
    T1.set_savepoint()
    T1.add_edge(2, 3)
    T1.restore_to_savepoint()
    edges_after = set(T1.edges())
    if len(edges_before) == len(edges_after):
        print("Critical Test Passed.")
    else:
        print("Critical Test Failed.")

    # print(T1.neighbors(1))
    # print(T1.neighbors(2))
    T2 = DirectedGraphData.FromNetworkX(G)
    print(T2.is_directed())
    print(T2['weight'][(1, 2)])
    print(T2['weight'][(2, 1)])
    print(T2.neighbors(1))
    print(T2.neighbors(2))
    print(T2.out_neighbors(1))
    print(T2.out_neighbors(2))
    print(T2.in_neighbors(1))
    print(T2.in_neighbors(2))
    T1.add_trait("hola")
    T1['hola'][(3, 2)] = "wassup?"
    print(T1["hola"][(2, 3)])
    print(T1.nodes())
    print(T2.nodes())
    print("---")
    T1_C = T1.copy()
    T2_C = T2.copy()
    T1_C.add_edge(1, 3)
    T2_C.add_edge(1, 3)
    print(T1.__edges__)
    print(T1_C.__edges__)
    print(T2.__edges__)
    print(T2_C.__edges__)
    T1_C["weight"][(1, 2)] = "changed"
    T1["weight"][(2, 1)] = "old"
    print("%s vs %s" % (T1_C["weight"][(1, 2)], T1["weight"][(1, 2)]))

    print("The Critical Test For Directed:")
    edges_before = set(T2.edges())
    T2.set_savepoint()
    T2.add_edge(1, 2)
    T2.restore_to_savepoint()
    edges_after = set(T2.edges())
    if len(edges_before) == len(edges_after):
        print("Critical Test Passed.")
    else:
        print("Critical Test Failed.")
