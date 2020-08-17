from graph_change import GraphChange
from graph_change_feature_counts import GraphChangeFeatureCounter
from trait_updater import TraitUpdater

__NODE_DEGREE_TRAIT_NAME__ = "Node Degree"
__INVOLVED_NODE_DEGREE_TRAIT_NAME__ = "InvNode Degree"

def NodeDegreeTrait():
    return GraphChangeFeatureCounter.RankTrait(__NODE_DEGREE_TRAIT_NAME__, \
        guaranteed_unique=False)

def InvolvedNodeDegreeTrait():
    return GraphChangeFeatureCounter.ClassTrait(\
        __INVOLVED_NODE_DEGREE_TRAIT_NAME__, \
        ["Higher", "Lower", "Equal", None])

class NodeDegreeTraitUpdater(TraitUpdater):

    # `graph_data` should be an object of type GraphData or type
    #   DirectedGraphData upon which the changes are occurring.
    def __init__(self, graph_data):
        self.__name__ = __NODE_DEGREE_TRAIT_NAME__
        self.__graph_data__ = graph_data
        self.__graph_data__.add_trait(self.__name__)
        for node in self.__graph_data__.nodes():
            self.__graph_data__[self.__name__][node] = \
                len(self.__graph_data__.neighbors(node))

    # `changes_occurred` is a list of elements of the GraphChange subclasses.
    #   Each element in `changes_occurred` should have the same timestamp.
    def update_before_changes(self, changes_occurred):
        pass

    # `changes_occurred` is a list of elements of the GraphChange subclasses.
    #   Each element in `changes_occurred` should have the same timestamp.
    def update_after_changes(self, changes_occurred):
        pass

    # `change` is a single element of a GraphChange subclass.
    def update_just_before_change_labeled(self, change):
        pass

    # `change` is a single element of a GraphChange subclass.
    def update_just_after_change_labeled(self, change):
        nodes = change.nodes_affected_after()

        for node in nodes:
            self.__graph_data__[self.__name__][node] = \
                len(self.__graph_data__.neighbors(node))

class InvolvedNodeDegreeTraitUpdater(TraitUpdater):

    # `graph_data` should be an object of type GraphData or type
    #   DirectedGraphData upon which the changes are occurring.
    def __init__(self, graph_data):
        self.__name__ = __INVOLVED_NODE_DEGREE_TRAIT_NAME__
        self.__graph_data__ = graph_data
        self.__graph_data__.add_trait(self.__name__)
        for node in self.__graph_data__.nodes():
            self.__graph_data__[self.__name__][node] = None

    # `changes_occurred` is a list of elements of the GraphChange subclasses.
    #   Each element in `changes_occurred` should have the same timestamp.
    def update_before_changes(self, changes_occurred):
        pass

    # `changes_occurred` is a list of elements of the GraphChange subclasses.
    #   Each element in `changes_occurred` should have the same timestamp.
    def update_after_changes(self, changes_occurred):
        pass

    # `change` is a single element of a GraphChange subclass.
    def update_just_before_change_labeled(self, change):
        if change.get_type() == GraphChange.NODE_DELETION:
            raise ValueError("Error! InvolvedNodeDegreeTraitUpdater not " + \
                "defined for node deletions.")
        nodes = change.nodes_affected_before()
        assert len(nodes) == 2
        degrees = [len(self.__graph_data__.neighbors(node)) for node in nodes]
        if degrees[0] == degrees[1]:
            values = ["Equal", "Equal"]
        elif degrees[0] > degrees[1]:
            values = ["Higher", "Lower"]
        else:
            values = ["Lower", "Higher"]

        for node_idx in range(0, len(nodes)):
            node = nodes[node_idx]
            value = values[node_idx]
            self.__graph_data__[self.__name__][node] = value

    # `change` is a single element of a GraphChange subclass.
    def update_just_after_change_labeled(self, change):
        nodes = change.nodes_affected_after()
        for node in nodes:
            self.__graph_data__[self.__name__][node] = None
