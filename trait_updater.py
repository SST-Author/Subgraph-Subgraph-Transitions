# The TraitUpdater class can update values for traits in a graph_data object at
#   four different instances:
#
#   1. Before all changes with a given timestamp are labeled (but after they
#       are executed IFF not using apply_addi(dele)tions_while_labeling).
#   2. Just before a change is labeled with surrounding subgraphs
#   3. Just after a change is labeled with surrounding subgraphs
#   4. After all changes with a given timestamp are executed

class TraitUpdater:

    # `graph_data` should be an object of type GraphData or type
    #   DirectedGraphData upon which the changes are occurring.
    def __init__(self, graph_data):
        pass

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
        pass

# Does nothing to update traits.
class NonUpdater(TraitUpdater):
    pass
