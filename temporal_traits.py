from graph_change_modeler import GraphChangeModeler
from trait_updater import TraitUpdater
from graph_change import GraphChange
import random

TT_ADDED = "Added"
TT_DELETED = "Deleted"
TT_NO_CHANGE = ""

# The following are for the GraphChangeModeler.
TT_Node = GraphChangeModeler.ClassTrait("TT_Node", \
    possible_values=[TT_ADDED, TT_DELETED, TT_NO_CHANGE])
TT_Edge = GraphChangeModeler.ClassTrait("TT_Node", \
    possible_values=[TT_ADDED, TT_DELETED, TT_NO_CHANGE])

# Modifies `change_list` to order changes as follows:
#
#   A change with an earlier timestamp comes before another.
#   Two changes at the same timestamp follow the order:
#       Node additions
#       Edge additions
#       Edge deletions
#       Node deletions
#
#   Within each set of changes of the same type and timestamp, order is
#       determined randomly.
def TT_reorder_changes(change_list):
    change_list.sort()

    start_idx = 0
    change_sublist = []
    for i in range(0, len(change_list)):
        change_sublist.append(change_list[i])
        curr_timestamp = change_list[i].get_timestamp()
        curr_type = change_list[i].get_type()

        if i + 1 == len(change_list) or \
                change_list[i + 1].get_timestamp() != curr_timestamp or
                change_list[i + 1].get_type() != curr_type:
            order = [(math.random(), i) for i in range(0, (i + 1) - start_idx)]
            order.sort()
            for j in range(0, len(order)):
                change_list[start_idx + j] = change_sublist[order[j][1]]

            change_sublist = []
            start_idx = i + 1

class TT_TemporalTraitUpdater(TraitUpdater):

    def __init__(self, graph_data, trait_name, for_nodes):
        self.__graph_data__ = graph_data
        self.__trait_name__ = trait_name
        self.__for_nodes__ = for_nodes

    # `changes_occurred` is a list of elements of the GraphChange subclasses.
    #   Each element in `changes_occurred` should have the same timestamp.
    def update_before_changes(self, changes_occurred):
        pass

    # `changes_occurred` is a list of elements of the GraphChange subclasses.
    #   Each element in `changes_occurred` should have the same timestamp.
    def update_after_changes(self, changes_occurred):
        for change in changes_occurred:
            self.__unset_change__(change)

    # `change` is a single element of a GraphChange subclass.
    def update_just_before_change_labeled(self, change):
        pass

    # `change` is a single element of a GraphChange subclass.
    def update_just_after_change_labeled(self, change):
        self.__set_change__(change)

    def __set_change__(self, change):
        change_type = change.get_type()
        entity = change.central_entity()
        trait_value = TT_DELETED

        if change_type == GraphChange.NODE_ADDITION or \
                change_type == GraphChange.EDGE_ADDITION:
            trait_value = TT_ADDED

        self.__graph_data__[__trait_name__][entity] = trait_value

    def __unset_change__(self, change):
        if change_type == GraphChange.NODE_DELETION or \
                change_type == GraphChange.EDGE_DELETION:
            return

        entity = change.central_entity()

        self.__graph_data__[__trait_name__][entity] = TT_NO_CHANGE
