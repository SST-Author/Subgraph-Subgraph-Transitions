"""
Synthetic temporal graph generators - only Barabasi Albert for the time being
"""
import random
from sys import argv
import networkx as nx


def _random_subset(seq, m, seed):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    random.seed(seed)

    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets


def barabasi_albert_graph(T, m, seed=None, is_directed=False):
    """
    Generates and writes a temporal BA graph
    T: number of timestamps
    m: number of edges the new node connects to
    """
    n = m + T - 1
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Add m initial nodes (m0 in barabasi-speak)
    time = 0
    starting_graph = nx.path_graph(m)  # start with a tree
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_edges_from(starting_graph.edges(data=True), t=time, w=1)

    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        time += 1  # increase time
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets), t=time, w=1)
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1

    edgelist_path = f'datasets/synthetic/BA_{m}_{T}_raw.g'
    print(f'Weighted edgelist written at {edgelist_path!r}')
    nx.write_edgelist(G, path=edgelist_path, data='t')
    return G

if __name__ == '__main__':
    if len(argv) < 3:
        print('Enter no of edges and no of time stamps')
        exit(1)
    m = int(argv[1])
    T = int(argv[2])
    g = barabasi_albert_graph(m=m, T=T, is_directed=True)

