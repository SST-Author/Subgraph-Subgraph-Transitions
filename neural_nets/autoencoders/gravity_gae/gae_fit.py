from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from collections import namedtuple

from neural_nets.autoencoders.gravity_gae.evaluation import compute_scores
#from gravity_gae.input_data import load_data
from neural_nets.autoencoders.gravity_gae.model import *
from neural_nets.autoencoders.gravity_gae.optimizer import OptimizerAE, OptimizerVAE
from neural_nets.autoencoders.gravity_gae.preprocessing import *

from neural_nets.utils import get_prob_mat_from_emb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

flags = namedtuple('FLAGS', ['dataset', 'task', 'model', 'dropout', 'epochs', 'features', 'lamb', 'learning_rate', 'hidden', 'dimension', 'normalize', 'epsilon', 'nb_run', 'prop_val', 'prop_test', 'validation', 'verbose', 'kcore', 'k', 'nb_iterations'])
FLAGS = flags('custom', 'link_prediction', 'gravity_gcn_ae', 0., 200, False, 1., 0.1, 64, 32, False, 0.01, 1, 5., 10., False, True, False, 2, 10)

def fit_model(adj, val_edges, val_edges_false, test_edges, test_edges_false, model_name):
    # Lists to collect average results
    mean_roc = []
    mean_ap = []
    mean_time = []

    # Load graph dataset
    #if FLAGS.verbose:
    #    print("Loading data...")
    #adj_init, features = load_data(FLAGS.dataset)
    print(f'Loading data... n: {adj.shape[0]}, m: {adj.nnz//2}')

    # The entire training process is repeated FLAGS.nb_run times
    for i in range(FLAGS.nb_run):
        # Start computation of running times
        t_start = time.time()

        # Preprocessing and initialization
        if FLAGS.verbose:
            print("Preprocessing and Initializing...")
        # Compute number of nodes
        num_nodes = adj.shape[0]
        # If features are not used, replace feature matrix by identity matrix
        if not FLAGS.features:
            features = sp.identity(adj.shape[0])
        # Preprocessing on node features
        features = sparse_to_tuple(features)
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Define placeholders
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        # Create model
        model = None
        model_name = model_name.lower()
        if model_name == 'gcn_ae':
            # Standard Graph Autoencoder
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_name == 'gcn_vae':
            # Standard Graph Variational Autoencoder
            model = GCNModelVAE(placeholders, num_features, num_nodes,
                                features_nonzero)
        elif model_name == 'source_target_gcn_ae':
            # Source-Target Graph Autoencoder
            if FLAGS.dimension % 2 != 0:
                raise ValueError('Dimension must be even for Source-Target models')
            model = SourceTargetGCNModelAE(placeholders, num_features,
                                        features_nonzero)
        elif model_name == 'source_target_gcn_vae':
            # Source-Target Graph Variational Autoencoder
            if FLAGS.dimension % 2 != 0:
                raise ValueError('Dimension must be even for Source-Target models')
            model = SourceTargetGCNModelVAE(placeholders, num_features,
                                        num_nodes, features_nonzero)
        elif model_name == 'gravity_gcn_ae':
            # Gravity-Inspired Graph Autoencoder
            model = GravityGCNModelAE(placeholders, num_features,
                                    features_nonzero)
        elif model_name == 'gravity_gcn_vae':
            # Gravity-Inspired Graph Variational Autoencoder
            model = GravityGCNModelVAE(placeholders, num_features, num_nodes,
                                    features_nonzero)
        else:
            raise ValueError('Undefined model!')

        # Optimizer (see tkipf/gae original GAE repository for details)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        with tf.name_scope('optimizer'):
            # Optimizer for Non-Variational Autoencoders
            if model_name in ('gcn_ae', 'source_target_gcn_ae', 'gravity_gcn_ae'):
                opt = OptimizerAE(preds = model.reconstructions,
                                labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                validate_indices = False), [-1]),
                                pos_weight = pos_weight,
                                norm = norm)
            # Optimizer for Variational Autoencoders
            elif model_name in ('gcn_vae', 'source_target_gcn_vae', 'gravity_gcn_vae'):
                opt = OptimizerVAE(preds = model.reconstructions,
                                labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                validate_indices = False), [-1]),
                                model = model,
                                num_nodes = num_nodes,
                                pos_weight = pos_weight,
                                norm = norm)

        # Normalization and preprocessing on adjacency matrix
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

        # Initialize TF session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Model training
        print(f"Training {model_name}...")

        t = time.time()
        print_every = 50
    ## Flag to compute total running time
    #t_start = time.time()
        for epoch in range(FLAGS.epochs + 1):
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                            feed_dict = feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            if epoch > 0 and epoch % print_every == 0 and FLAGS.verbose:
                # Display epoch information
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                    "time=", "{:.5f}".format(time.time() - t))
                # Validation (implemented for Task 1 only)
                if FLAGS.validation and FLAGS.task == 'link_prediction':
                    feed_dict.update({placeholders['dropout']: 0})
                    emb = sess.run(model.z_mean, feed_dict = feed_dict)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    val_roc, val_ap = compute_scores(val_edges, val_edges_false, emb)
                    print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))
        # Flag to compute Graph AE/VAE training time
        t_model = time.time()

        # Get embedding from model
        emb = sess.run(model.z_mean, feed_dict = feed_dict)

        # Test model
        print("Testing model...")
        if FLAGS.task == 'link_prediction':
            # Compute ROC and AP scores on test sets
            roc_score, ap_score = compute_scores(test_edges, test_edges_false, emb)
            # Append to list of scores over all runs
            mean_roc.append(roc_score)
            mean_ap.append(ap_score)

        sess.close()    # close the TensorFlow session and free up resources

        prob_mat = get_prob_mat_from_emb(emb)
        return prob_mat

## Report final results
#print("\nTest results for", FLAGS.model,
#      "model on", FLAGS.dataset, "on", FLAGS.task, "\n",
#      "___________________________________________________\n")
#
#print("AUC scores\n", mean_roc)
#print("Mean AUC score: ", np.mean(mean_roc),
#      "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")
#
#print("AP scores \n", mean_ap)
#print("Mean AP score: ", np.mean(mean_ap),
#      "\nStd of AP scores: ", np.std(mean_ap), "\n \n")
#
#print("Running times\n", mean_time)
#print("Mean running time: ", np.mean(mean_time),
#      "\nStd of running time: ", np.std(mean_time), "\n \n")
