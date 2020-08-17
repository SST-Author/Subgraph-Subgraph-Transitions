import random

# Interface for predicting links.
class StaticLinkPredictor(object):

    def __init__(self, graph_nodes, graph_edges, directed=False):
        pass

    # Given a list of edges, return a list of scores.
    def score_edges(self, edges):
        pass

class RandomLinkPredictor(StaticLinkPredictor):
    def __init__(self, graph_nodes, graph_edges, directed=False):
        pass

    def score_edges(self, edges):
        return [random.random() for edge in edges]

class CommonNeighborsStaticLinkPredictor(StaticLinkPredictor):
    def __init__(self, graph_nodes, graph_edges, directed=False):
        if directed:
            self.out_neighbors = {n: set() for n in graph_nodes}
            self.in_neighbors = {n: set() for n in graph_nodes}
        else:
            self.neighbors = {n: set() for n in graph_nodes}
        for (a, b) in graph_edges:
            if directed:
                self.out_neighbors[a].add(b)
                self.in_neighbors[b].add(a)
            else:
                self.neighbors[a].add(b)
                self.neighbors[b].add(a)
        self.directed = directed

    def score_edges(self, edges):
        scores = []
        for (a, b) in edges:
            if self.directed:
                scores.append(\
                    len(self.out_neighbors[a] & self.in_neighbors[b]) + \
                    len(self.out_neighbors[a] & self.out_neighbors[b]) + \
                    len(self.in_neighbors[a] & self.in_neighbors[b]) + \
                    len(self.in_neighbors[a] & self.out_neighbors[b]))
            else:
                scores.append(len(self.neighbors[a] & self.neighbors[b]))
        return scores

# Expects input edges to be 4-tuples (u, v, timestamp, weight)
#
# This is for predicting temporary links, which effectively represent
#   "interactions" and can be repeated accross timesteps.
class TemporalLinkPredictor(object):
    def __init__(self, graph_nodes, graph_edges, directed=False):
        pass

    # Given a list of edges, return a list of scores.
    def score_edges(self, edges):
        pass

class CommonNeighborsTemporalLinkPredictor(TemporalLinkPredictor):
    def __init__(self, graph_nodes, graph_edges, directed=False):
        if directed:
            self.out_neighbors = {n: set() for n in graph_nodes}
            self.in_neighbors = {n: set() for n in graph_nodes}
        else:
            self.neighbors = {n: set() for n in graph_nodes}
        for (a, b, _, _) in graph_edges:
            if directed:
                self.out_neighbors[a].add(b)
                self.in_neighbors[b].add(a)
            else:
                self.neighbors[a].add(b)
                self.neighbors[b].add(a)
        self.directed = directed

    def score_edges(self, edges):
        scores = []
        for (a, b, _) in edges:
            if self.directed:
                scores.append(\
                    len(self.out_neighbors[a] & self.in_neighbors[b]) + \
                    len(self.out_neighbors[a] & self.out_neighbors[b]) + \
                    len(self.in_neighbors[a] & self.in_neighbors[b]) + \
                    len(self.in_neighbors[a] & self.out_neighbors[b]))
            else:
                scores.append(len(self.neighbors[a] & self.neighbors[b]))
        return scores

class TGNLinkPredictor(TemporalLinkPredictor):
    def __init__(self, tgn, neg_sampler, directed=True):
        self.tgn = tgn
        self.neg_sampler = neg_sampler

    def score_edges(self, edges):
        from neural_nets.tgn.evaluation.evaluation import sst_score_edges
        scores, _ = sst_score_edges(self.tgn, self.neg_sampler, edges)
        return scores

class MatrixLinkPredictor(StaticLinkPredictor):

    def __init__(self, matrix, nodes):
        self.__matrix__ = matrix

    def score_edges(self, edges):
        scores = []
        for (u, v) in edges:
            scores.append(self.__matrix__[u,v])
        return scores
