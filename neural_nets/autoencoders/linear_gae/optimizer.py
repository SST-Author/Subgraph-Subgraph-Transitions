from collections import namedtuple

import tensorflow as tf

flags = namedtuple('FLAGS', ['dataset', 'task', 'model', 'dropout', 'epochs', 'features', 'learning_rate',
                             'hidden', 'dimension', 'nb_run', 'prop_val', 'prop_test', 'validation', 'verbose', 'kcore',
                             'k', 'nb_iterations'])

FLAGS = flags('custom', 'link_prediction', 'gcn_ae', 0., 200, False, 0.01, 32, 16, 1, 5., 10., False, True, False, 2,
              10)

'''
Disclaimer: the OptimizerAE and OptimizerVAE classes from this file
come from tkipf/gae original repository on Graph Autoencoders
'''


class OptimizerAE(object):
    """ Optimizer for non-variational autoencoders """
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     targets = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                     tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    """ Optimizer for variational autoencoders """
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     targets = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model.z_log_std \
                                               - tf.square(model.z_mean) \
                                               - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                              tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))