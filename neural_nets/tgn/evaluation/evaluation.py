import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)

def sst_score_edges(model, negative_edge_sampler, edges, n_neighbors=10, batch_size=200):
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  sources = np.asarray([u for (u, v, ts, idx) in edges])
  destinations = np.asarray([v for (u, v, ts, idx) in edges])
  timestamps = np.asarray([ts for (u, v, ts, idx) in edges])
  idxs = np.asarray([idx for (u, v, ts, idx) in edges])

  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    pred_scores = []
    true_labels = []
    for k in range(num_test_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
        sources_batch = sources[s_idx:e_idx]
        destinations_batch = destinations[s_idx:e_idx]
        timestamps_batch = timestamps[s_idx:e_idx]
        edge_idxs_batch = idxs[s_idx: e_idx]

        size = len(sources_batch)
        _, negative_samples = negative_edge_sampler.sample(size)

        pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

        pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
        true_label = np.concatenate([np.ones(size), np.asarray([])])

        pred_scores.extend([p.item() for p in pred_score[:size]])
        true_labels.extend([p.item() for p in true_label[:size]])

    #pred_scores = [p.item() for p in pred_score]
    #true_labels = [p.item() for p in true_label]

  return pred_scores, true_labels
