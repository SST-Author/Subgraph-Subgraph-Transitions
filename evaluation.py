from graph_change import *
from graph_data import GraphData, DirectedGraphData
import pickle as pkl
import math
import networkx as nx
import numpy as np
import os
import random
from sklearn import metrics, decomposition

def score_link_predictor(link_predictor, test_true_edges, test_false_edges, \
        aupr_extras, graph_name, directed, temporal, idx, model):

    scores, labels = get_score_and_label_vectors(\
        link_predictor, test_true_edges, test_false_edges, model)

    auc = metrics.roc_auc_score(labels, scores)
    fprs, tprs, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

    result = auc_values_for_curve_points(tprs, fprs)
    print("auc: %f" % auc)
    save_result("auc", result, graph_name, temporal, directed, idx, model)

    for (k, (_, _, aupr_test_true_edges, aupr_test_false_edges)) in aupr_extras:
        assert len(aupr_test_true_edges) > 0
        assert len(aupr_test_false_edges) > 0

        scores, labels = get_score_and_label_vectors(\
            link_predictor, aupr_test_true_edges, aupr_test_false_edges, model)

        precisions, recalls, thresholds = \
            metrics.precision_recall_curve(labels, scores, pos_label=1)

        aupr = aupr_values_for_curve_points(precisions, recalls, \
            N = len(aupr_test_false_edges), P = len(aupr_test_true_edges))
        print("aupr at k=%d" % k)
        print({'proper_aupr': aupr['proper_aupr'], 'trapezoid_aupr': \
            aupr['trapezoid_aupr'], 'generous_aupr': aupr['generous_aupr'], \
            'stingy_aupr': aupr['stingy_aupr']})
        save_result("aupr_at_k%d" % k, aupr, graph_name, temporal, directed, \
            idx, model)

def get_score_and_label_vectors(link_predictor, true_edges, false_edges, model=''):
    if model.lower() == 'tgn' or model.lower() == 'temporalgraphnetwork':
        pass
    else:
        if len(true_edges[0]) == 4:
            true_edges = [(a, b, t) for (a, b, t, w) in true_edges]
            false_edges = [(a, b, t) for (a, b, t, w) in false_edges]

    true_scores = link_predictor.score_edges(true_edges)
    false_scores = link_predictor.score_edges(false_edges)

    labels = [1 for cs in true_edges] + [0 for ns in false_edges]
    scores = true_scores + false_scores

    # Randomize order so that ties don't give priority to correct or
    #   incorrect in precision computation.
    perm = np.random.permutation([i for i in range(0, len(scores))])
    assert len(labels) == len(scores)
    scores = [scores[perm[i]] for i in range(0, len(scores))]
    labels = [labels[perm[i]] for i in range(0, len(labels))]

    return scores, labels

def aupr_values_for_curve_points(precisions, recalls, N, P):
    proper_aupr = 0.0
    trapezoid_aupr = 0.0
    generous_aupr = 0.0
    stingy_aupr = 0.0
    # Calculated from right to left!
    for j in range(1, len(precisions)):
        # Decreasing recall (right to left).
        r0 = recalls[j - 1]
        r1 = recalls[j]
        # Increasing precision (bottom to top).
        p0 = precisions[j - 1]
        p1 = precisions[j]
        trapezoid_aupr += (r0 - r1) * ((p0 + p1) / 2.0)
        generous_aupr += (r0 - r1) * max(p0, p1)
        stingy_aupr += (r0 - r1) * min(p0, p1)
        assert (r0 - r1) >= 0
        if r0 != r1:
            # Uses the solved integral of area under two PR points obtained
            #   via a conversion to ROC space with linear interpolation back
            #   into PR space.
            tpr_0 = r0
            tpr_1 = r1
            tp_0 = tpr_0 * P
            tp_1 = tpr_1 * P
            if p0 == 0.0:
                fp_0 = float(N)
            else:
                fp_0 = (tp_0 / p0) - tp_0
            if p1 == 0.0:
                fp_1 = float(N)
            else:
                fp_1 = (tp_1 / p1) - tp_1
            fpr_0 = fp_0 / N
            fpr_1 = fp_1 / N
            A = P * (tpr_0 - tpr_1)
            B = P * tpr_1
            C = P * (tpr_0 - tpr_1) + N * (fpr_0 - fpr_1)
            if C == 0.0:
                print("Info: Unexpected C == 0.0 in AUPR calculation " + \
                    "(evaluation.py)")
                continue
            D = P * tpr_1 + N * fpr_1
            part_one = A / C
            if D > 0.0:
                part_two = (1.0 / C) * (B - (A*D) / C) * math.log((C+D) / D)
            else:
                part_two = 0.0
            area = (tpr_0 - tpr_1) * (part_one + part_two)
            proper_aupr += area

    values_dict = {'proper_aupr': proper_aupr, 'generous_aupr': generous_aupr, \
            'stingy_aupr': stingy_aupr, 'trapezoid_aupr': trapezoid_aupr, \
            'recalls': recalls, 'precisions': precisions}

    return values_dict

    aupr = 0.0
    generous_aupr = 0.0
    stingy_aupr = 0.0
    # Calculated from right to left!
    for j in range(1, len(precisions)):
        # Decreasing recall (right to left).
        x0 = recalls[j - 1]
        x1 = recalls[j]
        # Increasing precision (bottom to top).
        y0 = precisions[j - 1]
        y1 = precisions[j]
        aupr += (x0 - x1) * ((y0 + y1) / 2.0)
        generous_aupr += (x0 - x1) * max(y0, y1)
        stingy_aupr += (x0 - x1) * min(y0, y1)
        assert (x0 - x1) >= 0

    return {'aupr': aupr, 'generous_aupr': generous_aupr, \
        'stingy_aupr': stingy_aupr, \
        'recalls': recalls, 'precisions': precisions}

def auc_values_for_curve_points(tprs, fprs):
    auc = 0.0
    generous_auc = 0.0
    stingy_auc = 0.0
    # Calculated from left to right!
    for j in range(1, len(fprs)):
        # Increasing fpr
        x0 = fprs[j - 1]
        x1 = fprs[j]
        # Increasing true positive rate.
        y0 = tprs[j - 1]
        y1 = tprs[j]
        auc += (x1 - x0) * ((y0 + y1) / 2.0)
        generous_auc += (x1 - x0) * max(y0, y1)
        stingy_auc += (x1 - x0) * min(y0, y1)
        assert (x1 - x0) >= 0

    return {'auc': auc, 'generous_auc': generous_auc, \
        'stingy_auc': stingy_auc, \
        'fprs': fprs, 'tprs': tprs}

def save_result(result_name, result_value, graph_name, temporal, directed, idx,\
        model):
    directed_str = ["undirected", "directed"][int(directed)]
    temporal_str = ["static", "temporal"][int(temporal)]
    path = f'test_results/{temporal_str}/{graph_name}_{directed_str}/{model}_{result_name}_{idx}.pkl'
    __prepare_path__(path)
    f = open(path, "wb")
    pkl.dump(result_value, f)
    f.close()

def save_model_data(model_data, data_name, model, graph_name, temporal, directed, idx):
    directed_str = ["undirected", "directed"][int(directed)]
    temporal_str = ["static", "temporal"][int(temporal)]
    path = f'model_data/{temporal_str}/{graph_name}_{directed_str}/{model}_{data_name}_{idx}.pkl'
    __prepare_path__(path)
    f = open(path, "wb")
    pkl.dump(model_data, f)
    f.close()

def load_model_data(data_name, model, graph_name, temporal, directed, idx):
    directed_str = ["undirected", "directed"][int(directed)]
    temporal_str = ["static", "temporal"][int(temporal)]
    path = f'model_data/{temporal_str}/{graph_name}_{directed_str}/{model}_{data_name}_{idx}.pkl'
    f = open(path, "rb")
    model_data = pkl.load(f)
    f.close()
    return model_data

def __prepare_path__(filepath, overwrite=True):
    if os.path.isfile(filepath):
        return overwrite
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return True
