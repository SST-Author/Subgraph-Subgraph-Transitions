
# The GraphChange class and its subclasses, EdgeAddition, EdgeDeletion,
#   NodeAddition, and NodeDeletion, are used to model changes on the
#   GraphData and DirectedGraphData classes. Each change has four methods
#   available:
#
#   * perform() -- executes the change on `graph_data` -- note that
#       for node deletion, the class object saves information on which nodes
#       the deleted node was connected to so that the change can be undone.
#
#   * undo() -- undoes the change on `graph_data` -- note that for
#       undoing node deletion, this assumes that all the nodes the deleted
#       node was connected to are still present in the graph.
#
#   * get_type() -- returns one of the following:
#       GraphChange.EDGE_ADDITION
#       GraphChange.EDGE_DELETION
#       GraphChange.NODE_ADDITION
#       GraphChange.NODE_DELETION   
#
#   * get_timestamp() -- returns the timestamp.
#
#   * set_timestamp(new_timestamp) -- allows the timestamp to be changed.
#
#   * get_permanent() -- returns the change's "permanent" flag -- useful for
#       trait updaters to know how to handle changes
#
#   * set_permanent(is_permanent) -- allows the "permanent" flag to be changed.
#
#   * nodes_affected_before() -- returns all the nodes in the graph before the
#       change that are in some way modified -- for example, if a node X was
#       added to connect to Y, the result would just be the node [Y].
#
#   * nodes_affected_after() -- returns all the nodes in the graph after the
#       change that are in some way affected -- for example, if a node X was
#       added to connect to Y, the result would be the nodes [X, Y].
#
#   * central_entity() -- returns the node or edge being added or deleted
class GraphChange:

    def __init__(self):
        self.type = None
        self.timestamp = None

    # Note that the ordering of these values is relevant to the __lt__ method.
    NODE_ADDITION = 0
    EDGE_ADDITION = 1
    EDGE_DELETION = 2
    NODE_DELETION = 3

    def perform(self):
        raise TypeError("Error! Called perform() on base class GraphChange.")

    def undo(self):
        raise TypeError("Error! Called undo() on base class GraphChange.")

    def get_type(self):
        return self.type

    def get_timestamp(self):
        return self.timestamp

    def set_timestamp(self, new_timestamp):
        self.timestamp = new_timestamp

    def get_permanent(self):
        return self.permanent

    def set_permanent(self, is_permanent):
        self.permanent = is_permanent

    def nodes_affected_before(self):
        raise TypeError("Error! Called nodes_affected_before() on base " + \
            "class GraphChange.")

    def nodes_affected_after(self):
        raise TypeError("Error! Called nodes_affected_after() on base " + \
            "class GraphChange.")

    def edges_affected_before(self):
        raise TypeError("Error! Called edges_affected_before() on base " + \
            "class GraphChange.")

    def edges_affected_after(self):
        raise TypeError("Error! Called edges_affected_after() on base " + \
            "class GraphChange.")

    def central_entity(self):
        raise TypeError("Error! Called central_entity() on base " + \
            "class GraphChange.")

    def __lt__(self, other):
        return self.timestamp < other.timestamp or \
            (self.timestamp == other.timestamp and self.type < other.type)

class EdgeAddition(GraphChange):

    def __init__(self, graph_data, source, target, timestamp=0, permanent=True):
        self.type = GraphChange.EDGE_ADDITION
        self.graph_data = graph_data
        self.source = source
        self.target = target
        self.timestamp = timestamp
        self.permanent = permanent

    def perform(self):
        self.graph_data.add_edge(self.source, self.target)

    def undo(self):
        self.graph_data.delete_edge(self.source, self.target)

    def nodes_affected_before(self):
        return [self.source, self.target]

    def nodes_affected_after(self):
        return [self.source, self.target]

    def edges_affected_before(self):
        return []

    def edges_affected_after(self):
        return [(self.source, self.target)]

    def central_entity(self):
        return (self.source, self.target)

class EdgeDeletion(GraphChange):

    def __init__(self, graph_data, source, target, timestamp=0, permanent=True):
        self.type = GraphChange.EDGE_DELETION
        self.graph_data = graph_data
        self.source = source
        self.target = target
        self.timestamp = timestamp
        self.permanent = permanent

    def perform(self):
        self.graph_data.delete_edge(self.source, self.target)

    def undo(self):
        self.graph_data.add_edge(self.source, self.target)

    def nodes_affected_before(self):
        return [self.source, self.target]

    def nodes_affected_after(self):
        return [self.source, self.target]

    def edges_affected_before(self):
        return [(self.source, self.target)]

    def edges_affected_after(self):
        return []

    def central_entity(self):
        return (self.source, self.target)

class NodeAddition(GraphChange):

    # In the case of a node addition, the new node is assumed to connect to a
    #   single node.
    # If the graph is directed, a source can be specified, indicating which
    #   node points to which. If none is specified, the new node will be treated
    #   as the source.
    def __init__(self, graph_data, new_node, neighbor, source=None, \
            timestamp=0, permanent=True):
        self.type = GraphChange.NODE_ADDITION
        self.graph_data = graph_data
        self.new_node = new_node
        self.neighbor = neighbor
        if (source is not None) and source == neighbor:
            self.source = neighbor
            self.target = new_node
        else:
            self.source = new_node
            self.target = neighbor
        self.timestamp = timestamp

    def perform(self):
        self.graph_data.add_node(self.new_node)
        self.graph_data.add_edge(self.source, self.target)

    def undo(self):
        self.graph_data.delete_node(self.new_node)

    def nodes_affected_before(self):
        return [self.neighbor]

    def nodes_affected_after(self):
        return [self.new_node, self.neighbor]

    def edges_affected_before(self):
        return []

    def edges_affected_after(self):
        return [(self.source, self.target)]

    def central_entity(self):
        return self.new_node

    def neighbor(self):
        return self.neighbor

class NodeDeletion(GraphChange):

    def __init__(self, graph_data, node, timestamp=0, permanent=True):
        self.type = GraphChange.NODE_DELETION
        self.graph_data = graph_data
        self.node = node
        self.timestamp = timestamp
        self.performed = False

        # If node is in the graph, fill out the "affected" information.
        if self.graph_data.has_node(node):
            self.affected_before = list(self.graph_data.neighbors(self.node))+\
                [node]
            self.affected_after = list(self.graph_data.neighbors(self.node))
        else:
            self.affected_before = None
            self.affected_after = None

    def perform(self):
        # Update the "affected" data to reflect when the change actually
        #   occured.
        self.affected_before = list(self.graph_data.neighbors(self.node)) + \
            [self.node]
        self.affected_after = list(self.graph_data.neighbors(self.node))

        if self.graph_data.is_directed():
            self.out_neighbors = list(self.graph_data.out_neighbors(self.node))
            self.in_neighbors = list(self.graph_data.in_neighbors(self.node))
        else:
            self.neighbors = list(self.graph_data.neighbors(self.node))

        self.graph_data.delete_node(self.node)
        self.performed = False

    def undo(self):
        if not self.performed:
            raise ValueError("Error! Cannot undo() node deletion for node " + \
                "%s because deletion has not yet been " % (self.node) + \
                "perform()'ed - thus lacking neighbor information.")

        self.graph_data.add_node(self.node)

        if self.graph_data.is_directed():
            for out_n in self.out_neighbors:
                self.graph_data.add_edge(self.node, out_n)
            for in_n in self.in_neighbors:
                self.graph_data.add_edge(in_n, self.node)
        else:
            for n in self.neighbors:
                self.graph_data.add_edge(self.node, n)

    def nodes_affected_before(self):
        if self.affected_before is None:
            raise ValueError("Error! Cannot call nodes_affected_before() " + \
                "on a node deletion change without knowing the node's " + \
                "neighbors (node %s). Must perform() the " % (self.node) + \
                "change first or re-initialize on graph_data containing " + \
                "the node.")
        return list(self.affected_before)

    def nodes_affected_after(self):
        if self.affected_after is None:
            raise ValueError("Error! Cannot call nodes_affected_after() " + \
                "on a node deletion change without knowing the node's " + \
                "neighbors (node %s). Must perform() the " % (self.node) + \
                "change first or re-initialize on graph_data containing " + \
                "the node.")
        return list(self.affected_after)

    def edges_affected_before(self):
        if self.affected_before is None:
            raise ValueError("Error! Cannot call edges_affected_before() " + \
                "on a node deletion change without knowing the node's " + \
                "neighbors (node %s). Must perform() the " % (self.node) + \
                "change first or re-initialize on graph_data containing " + \
                "the node.")

        if self.__graph_data__.is_directed():
            return [(self.node, o_n) for o_n in self.out_neighbors] + \
                [(i_n, self.node) for i_n in self.in_neighbors]
        return [(self.node, n) for n in self.neighbors]

    def edges_affected_after(self):
        if self.affected_after is None:
            raise ValueError("Error! Cannot call edges_affected_after() " + \
                "on a node deletion change without knowing the node's " + \
                "neighbors (node %s). Must perform() the " % (self.node) + \
                "change first or re-initialize on graph_data containing " + \
                "the node.")
        return []

    def central_entity(self):
        return self.node
