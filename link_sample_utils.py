import networkx as nx
import random

def sample(elements, num_samples, with_replacement=True):
    print("Taking a sample of size %d from %d. Replacement=%s" % \
        (num_samples, len(elements), with_replacement))
    if not with_replacement:
        return random.sample(elements, num_samples)
    samples = []
    for i in range(0, num_samples):
        samples.append(random.choice(elements))
    return samples

# Returns a list of `num_samples` edges not in the graph's edge set.
def non_edges_sample(nodes, edges, directed, num_samples, with_replacement=True):
    num_nodes = len(nodes)
    num_edges = len(edges)
    nodes_list = list(nodes)
    nodes_list.sort()

    edges_set = edges
    if not (type(edges_set) is set):
        edges_set = set(edges_set)

    edges_list = list(edges)
    if len(edges_list[0]) == 3:
        edges_list = [(a, b) for (a, b, t) in edges]

    num_non_edges = int((num_nodes * (num_nodes - 1)) / \
        (2 - int(directed))) - num_edges

    print("Taking a non-edges sample of size %d from %d. Replacement=%s" % \
        (num_samples, num_non_edges, with_replacement))

    # If sufficiently large sample size,
    #   just enumerate all possible edges and sample.

    # Enumerate all possible non-edges only if sample size is a large fraction
    #   of non-edges.
    if num_samples > (num_non_edges * 0.75):
        possible_edges = set()
        if directed:
            for i in range(0, num_nodes):
                for j in range(0, num_nodes):
                    if j == i:
                        continue
                    possible_edges.add((nodes[i], nodes[j]))
        else:
            for i in range(0, num_nodes):
                for j in range(i + 1, num_nodes):
                    possible_edges.add((nodes[i], nodes[j]))
        non_edges = list(possible_edges - edges_set)
        if not with_replacement:
            return random.sample(non_edges, num_samples)
        result = []
        for i in range(0, num_samples):
            result.append(random.choice(non_edges))
        return result

    """ Skip the fancy stuff for now. It's SLOW and maybe wrong.
    if directed:
        num_possible_targets = {n: num_nodes - 1 for n in nodes_list}
    else:
        # If undirected, only consider one way of generating the edges
        #   (a, b) and (b, a): (min(a, b), max(a, b))
        num_possible_targets = {nodes_list[i]: (num_nodes - 1) - i \
            for i in range(0, num_nodes)}

    relevant_neighbors = {n: set() for n in nodes}

    for (a, b) in edges_list:
        if directed:
            num_possible_targets[a] -= 1
            relevant_neighbors[a].add(b)
        else:
            s = min(a, b)
            t = max(a, b)
            num_possible_targets[s] -= 1
            relevant_neighbors[s].add(t)

    total_non_edges = \
        float(sum([poss for n, poss in num_possible_targets.items()]))
    source_probs = [num_possible_targets[n] / total_non_edges for n in nodes_list]
    source_probs_cumulative = [source_probs[0]]
    s = source_probs[0]
    for i in range(1, num_nodes):
        s += source_probs[i]
        source_probs_cumulative.append(s)
    """

    if with_replacement:
        non_edges = []
    else:
        non_edges = set()

    while True:
        """ Skip the fancy stuff for now. It's SLOW and maybe wrong.
        source_idx = __cumulative_idx_select__(source_probs_cumulative)
        source = nodes[source_idx]
        if directed:
            valid_targets = []
            for n in nodes:
                if n not in relevant_neighbors[source]:
                    valid_targets.append(n)
        else:
            valid_targets = []
            for i in range(source_idx + 1, num_nodes):
                n = nodes[i]
                if n not in relevant_neighbors[source]:
                    valid_targets.append(n)

        target = valid_targets[random.randint(0, len(valid_targets) - 1)]
        """
        # Simple version. Way faster.
        source = random.choice(nodes_list)
        target = random.choice(nodes_list)
        if source == target or (source, target) in edges_set or \
                ((not directed) and (target, source) in edges_set):
            continue

        if not with_replacement and (source, target) in non_edges:
            continue

        if with_replacement:
            non_edges.append[(source, target)]
        else:
            non_edges.add((source, target))

        if len(non_edges) == num_samples:
            return list(non_edges)

# A binary search for the idx such that:
#
#   cp[idx - 1] < val <= cp[idx]
#   note the _strictly_ less than.
def __cumulative_idx_select__(cumulative_prob_list):
    val = random.random()
    num_options = len(cumulative_prob_list)
    upper = num_options  # exclusive
    lower = 0  # inclusive
    idx = int(upper / 2)
    while True:
        if lower + 1 == upper:
            return idx

        cp = cumulative_prob_list[idx]

        if val <= cp:
            if cumulative_prob_list[idx - 1] < val:
                return idx
            # otherwise, cumulative_prob_list[idx - 1] == val:
            upper = idx
        else:
            lower = idx + 1
        # i.e. overflow-safe idx = int((upper + lower) / 2)
        idx = lower + int((upper - lower) / 2)

def all_disconnected_node_pairs_within_k(graph, k):
    nodes = list(graph.nodes())
    nodes.sort()
    pairs = set()

    nx_graph = nx.Graph()
    for node in nodes:
        nx_graph.add_node(node)
    for (a, b) in graph.edges():
        nx_graph.add_edge(a, b)

    for node in nodes:
        bfs_edges = nx.bfs_edges(nx_graph, node, depth_limit=k)
        bfs_nodes = set([v for (u, v) in bfs_edges])
        if graph.is_directed():
            for bfs_n in bfs_nodes:
                if not graph.has_edge(node, bfs_n):
                    pairs.add((node, bfs_n))
        else:
            bfs_nodes = bfs_nodes - set(graph.neighbors(node))
            for bfs_n in bfs_nodes:
                if bfs_n > node:
                    pairs.add((node, bfs_n))
    return pairs

def all_connected_node_pairs_that_would_be_within_k_if_disconnected(graph, k):
    nodes = list(graph.nodes())
    nodes.sort()
    pairs = set()

    nx_graph = nx.Graph()
    for node in nodes:
        nx_graph.add_node(node)
    for (a, b) in graph.edges():
        nx_graph.add_edge(a, b)

    for node in nodes:
        neighbors = list(nx_graph.neighbors(node))
        for neighbor in neighbors:
            if neighbor > node:
                continue

            nx_graph.remove_edge(node, neighbor)

            k_half_a = int(k / 2)
            k_half_b = k - k_half_a
            bfs_edges_a = nx.bfs_edges(nx_graph, node, depth_limit=k_half_a)
            bfs_nodes_a = set([v for (u, v) in bfs_edges_a])
            bfs_edges_b = nx.bfs_edges(nx_graph, neighbor, depth_limit=k_half_b)
            bfs_nodes_b = set([v for (u, v) in bfs_edges_b])

            if len(bfs_nodes_a & bfs_nodes_b) > 0:
                if graph.is_directed():
                    if graph.has_edge(node, neighbor):
                        pairs.add((node, neighbor))
                    if graph.has_edge(neighbor, node):
                        pairs.add((neighbor, node))
                else:
                    pairs.add((node, neighbor))

            nx_graph.add_edge(node, neighbor)

    return pairs
