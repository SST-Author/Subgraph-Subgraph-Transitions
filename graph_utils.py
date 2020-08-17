
# Returns a list of sets containing the connected components of a graph.
# If the graph is directed, it treats the edges as undirected.
def connected_components(graph_data):
    # Remember, this set is not "safe" in that it's an internal variable for gd.
    nodes = graph_data.nodes()

    labels = {n: n for n in nodes}
    active = set(nodes)
    while len(active) > 0:
        a = active.pop()

        new_l = min([labels[a]] + [labels[n] for n in graph_data.neighbors(a)])
        if new_l < labels[a]:
            labels[a] = new_l
            for n in graph_data.neighbors(a):
                if labels[n] > new_l:
                    active.add(n)

    labels_list = sorted([(l, n) for n, l in labels.items()])
    components = []
    prev_label = None
    for (l, n) in labels_list:
        if l != prev_label:
            components.append([])
            prev_label = l
        components[-1].append(n)

    return components
