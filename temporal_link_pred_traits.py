from graph_change import GraphChange
from graph_change_feature_counts import GraphChangeFeatureCounter
from trait_updater import TraitUpdater

# Names of two traits.
TemporalLinkPredictionTraitNames = ["TLP: Freq", "TLP: Recency"]
# Names of 
TemporalLinkPredictionTraitNonValues = ["0", "Never"]
TemporalLinkPredictionTraits = [
    GraphChangeFeatureCounter.ClassTrait(\
        TemporalLinkPredictionTraitNames[0], \
            [TemporalLinkPredictionTraitNonValues[0], "1", "2", "3+"]), \
    GraphChangeFeatureCounter.ClassTrait(\
        TemporalLinkPredictionTraitNames[1], \
            [TemporalLinkPredictionTraitNonValues[1], "Newest", "New", "Old"])]
# ^^ Note: "0" and "Never" are only used on the edge presently being labeled,
#   and only if it was never in the graph before.

# Does not make use of weight information on timestamped edges. Does create
#   'weights' of its own for recording repetitions of an edge across the given
#   timestamp.

# This class updates a graph's values of both the above traits. If using both
#   traits in the GraphChangeFeatureCounter, pass:
#   edge_trait_updaters=[TemporalLinkPredictionUpdater(graph_data), NonUpdater(None)]
#
#   NonUpdater is a trait updater that does nothing. Adding it is required
#   because the GraphChangeFeatureCounter expects one updater per trait. This
#   updater should not be called twice.
#
# In the resulting SSTs, the listed trait value on the added edge will be the
#   trait value BEFORE the edge was added. It represents the value as of the
#   timestamp before the time at which the edge is added. Hence why a frequency
#   of "0" or a recency of "Never" are allowed -- they can apply to the edge being
#   added in the current SST.
class TemporalLinkPredictionTraitUpdater:

    def __init__(self, graph_data):
        self.__graph_data__ = graph_data
        assert self.__graph_data__.num_edges() == 0
        self.__freq__ = TemporalLinkPredictionTraitNames[0]
        self.__recency__ = TemporalLinkPredictionTraitNames[1]
        self.__graph_data__.add_trait(self.__freq__)
        self.__graph_data__.add_trait(self.__recency__)
        self.__curr_time__ = None

        self.__aging_map__ = {"Newest": "New", "New": "Old", "Old": "Old", \
                               "0": "1", "1": "2", "2": "3+", "3+": "3+"}

        self.__start_values__ = ["1", "Newest"]

        # Used to keep track of which edges were actually added to the graph and
        #   should have their value updated when we transition to the next timestep.
        self.__flagged_edges__ = set()

    # `changes_occurred` is a list of elements of the GraphChange class.
    #
    # The list is expected to be empty unless it contains EdgeAdditions.
    def update_before_changes(self, changes_occurred):
        # The list can be empty.
        if len(changes_occurred) == 0:
            return
        if changes_occurred[0].get_type() != GraphChange.EDGE_ADDITION:
            print("Warning: Running TemporalLinkPredictionTraitUpdater with " +\
                "changes other than Edge Additions has unknown consequences.")
            return
        if self.__curr_time__ is None or \
                changes_occurred[0].get_timestamp() != self.__curr_time__:
            # Moving on to another timestamp.

            self.__curr_time__ = changes_occurred[0].get_timestamp()

            for (u, v) in self.__graph_data__.edges():
                if (u, v) in self.__flagged_edges__:
                    # Update the edge to be in the "Newest" category.
                    self.__graph_data__[self.__recency__][(u, v)] = \
                        self.__start_values__[1]
                    # Increase the # of times this edge has occurred.
                    freq = self.__graph_data__[self.__freq__][(u, v)]
                    self.__graph_data__[self.__freq__][(u, v)] = \
                        self.__aging_map__[freq]
                else:
                    # The edge gets older.
                    recency = self.__graph_data__[self.__recency__][(u, v)]
                    self.__graph_data__[self.__recency__][(u, v)] = \
                        self.__aging_map__[recency]

            self.__flagged_edges__ = set()

    def update_after_changes(self, changes_occurred):
        if len(changes_occurred) == 0 or \
                changes_occurred[0].get_type() != GraphChange.EDGE_ADDITION:
            return
        edge_list = [c.central_entity() for c in changes_occurred]
        # print("Number of edges at this timestamp: %d" % len(edge_list))
        # print("Number of duplicate edges: %d" % (len(edge_list) - len(set(edge_list))))
        for change in changes_occurred:
            (u, v) = change.central_entity()
            if self.__graph_data__.has_edge(u, v):
                if change.get_permanent():
                    self.__flagged_edges__.add((u, v))
            else:
                if (u, v) in self.__graph_data__[self.__freq__]:
                    del self.__graph_data__[self.__freq__][(u, v)]
                    del self.__graph_data__[self.__recency__][(u, v)]

    def update_just_before_change_labeled(self, change):
        if change.get_type() != GraphChange.EDGE_ADDITION:
            return
        edge = change.central_entity()
        # If edge entirely new.
        if edge not in self.__graph_data__[self.__freq__]:
            self.__graph_data__[self.__freq__][edge] = \
                TemporalLinkPredictionTraitNonValues[0]
            self.__graph_data__[self.__recency__][edge] = \
                TemporalLinkPredictionTraitNonValues[1]

    def update_just_after_change_labeled(self, change):
        pass
