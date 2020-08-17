import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import namedtuple

from neural_nets.tgn.evaluation.evaluation import eval_edge_prediction
from neural_nets.tgn.model.tgn import TGN
from neural_nets.tgn.utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from neural_nets.tgn.utils.data_processing import get_data, compute_time_statistics
from neural_nets.tgn.utils.data_processing import Data

torch.manual_seed(0)
np.random.seed(0)

def create_data(full_data, train_data, val_data, test_data):
    return

def fit_model(full_data, nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, test_edges_false, max_idx, data_name='karate', n_runs=1, use_memory=True):
    # set up args
    args_template = namedtuple('args', ['data', 'bs', 'prefix', 'n_degree', 'n_head', 'n_epoch', 'n_layer', 'lr', 'patience', 'n_runs', 'drop_out', 'gpu', 'node_dim', 'time_dim', 'backprop_every', 'use_memory', 'embedding_module', 'message_function', 'aggregator', 'memory_update_at_end', 'message_dim', 'memory_dim', 'different_new_nodes', 'uniform'])
    args = args_template(data_name, 200, 'tgn-attn', 10, 2, 50, 1, 0.0001, 5, n_runs, 0.1, 0, 100, 100, 1, use_memory, 'graph_attention', 'identity', 'last', False, 100, 172, False, False)

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    SEQ_LEN = NUM_NEIGHBORS
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim

    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
    get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data \
            = get_data(full_data, nodes, train_edges, val_edges_true, val_edges_false, test_edges_true, test_edges_false, max_idx, different_new_nodes_between_val_and_test=args.different_new_nodes)

    ### Extract data for training, validation and testing

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                        seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                        new_node_test_data.destinations,
                                        seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
        results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
        Path("results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                    edge_features=edge_features, device=device,
                    n_layers=NUM_LAYER,
                    n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                    message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                    memory_update_at_start=not args.memory_update_at_end,
                    embedding_module_type=args.embedding_module,
                    message_function=args.message_function,
                    aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
                    mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                    mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
        tgn = tgn.to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)

        new_nodes_val_aps = []
        val_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if USE_MEMORY:
                tgn.memory.__init_memory__()

            # Train using only training graph
            tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            logger.info('start {} epoch'.format(epoch))
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(num_instance, start_idx + BATCH_SIZE)
                    sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                        train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]

                    size = len(sources_batch)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    with torch.no_grad():
                        pos_label = torch.ones(size, dtype=torch.float, device=device)
                        neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    tgn = tgn.train()
                    pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                        timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

                    loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if USE_MEMORY:
                    tgn.memory.detach_memory()

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            ### Validation
            # Validation uses the full graph
            tgn.set_neighbor_finder(full_ngh_finder)

            if USE_MEMORY:
                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
                train_memory_backup = tgn.memory.backup_memory()

            val_ap, val_auc = eval_edge_prediction(model=tgn, \
                                                   negative_edge_sampler=val_rand_sampler, \
                                                   data=val_data, \
                                                   n_neighbors=NUM_NEIGHBORS)
            if USE_MEMORY:
                val_memory_backup = tgn.memory.backup_memory()
                # Restore memory we had at the end of training to be used when validating on new nodes.
                # Also backup memory after validation so it can be used for testing (since test edges are
                # strictly later in time than validation edges)
                tgn.memory.restore_memory(train_memory_backup)

            # Validate on unseen nodes
            nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn, \
                                                         negative_edge_sampler=val_rand_sampler, \
                                                         data=new_node_val_data, \
                                                         n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
                # Restore memory we had at the end of validation
                tgn.memory.restore_memory(val_memory_backup)

            new_nodes_val_aps.append(nn_val_ap)
            val_aps.append(val_ap)
            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump({
                "val_aps": val_aps,
                "new_nodes_val_aps": new_nodes_val_aps,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
            logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

            # Early stopping
            if early_stopper.early_stop_check(val_ap):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                tgn.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                tgn.eval()
                break
            else:
                torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen nodes
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()

        ### Test
        tgn.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc = eval_edge_prediction(model=tgn, \
                                                 negative_edge_sampler=test_rand_sampler, \
                                                 data=test_data, \
                                                 n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)

        # Test on unseen nodes
        nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn, \
                                                       negative_edge_sampler=nn_test_rand_sampler, \
                                                       data=new_node_test_data, \
                                                       n_neighbors=NUM_NEIGHBORS)

        logger.info('Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
        logger.info('Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
        # Save results for this run
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "test_ap": test_ap,
            "new_node_test_ap": nn_test_ap,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving TGN model')
        if USE_MEMORY:
            # Restore memory at the end of validation (save a model which is ready for testing)
            tgn.memory.restore_memory(val_memory_backup)
        torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
        logger.info('TGN model saved')
    return tgn, test_rand_sampler
