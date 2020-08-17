## Splits of Graphs into Train, Validation, and Test sets.

These are splits of the graphs found in datasets/. Check out the readme there for information on the individual graphs.

Every graph's directory will have a `node_list.txt` file, along with 10 splits for evaluating Area Under ROC (i.e. AUC) and Area Under Precision Recall Curve (AUPR). In temporal graphs, the true edges are the same for all 10 splits and only the false edges change.

AUPR gets a different set of validation and test edges from AUC (but same train edges!) because unlike AUC, AUPR requires that negatives are not downsampled. This effectively means testing on ALL non-edges in the graph. That is typically too many edges to test on, so for AUPR we restrict the prediction task to be making predictions on edges (`u`, `v`) where the nodes are within a distance `k` of each other. Specifically, the true edges tested are the ones (`u`, `v`) that WOULD be within a distance `k` if the edge (`u`, `v`) were removed, and the false edges are the disconnected node pairs (`u`, `v`)  Hence you will see files with `k2` or `k3` on them; these are the files for AUPR (recall that the training files are the same for AUPR and AUC. It's just the evaluation that's different.)

`node_list.txt` -- just a list of nodes, one node per line

## Static

Each graph's edges is partitioned in an 85/5/10 train/validation/test split.

Edge files have one edge per line, where edge (`u`, `v`) is simply represented as `u` and `v` separated by a space character:
`u` `v`

## Temporal

Each graph's edges are partitioned earliest to last into 10 intervals (i.e. "buckets"). The original timestamps are replaced with bucket number. The first eight buckets are in the train files. The 9th is in the validation files, and the 10th is in the test files. It is possible that an edge occurs multiple times in the same bucket. Thus, each edge is given a "weight" value, which equals the number of times it appears in a bucket. Note that this means that the same edge can appear in different buckets with different weights.

Since partitioning of edges is not performed randomly but rather by the original timestamps, the training edges, true validation edges, and true test edges will be the same in every split; it is only the false edges that change. For AUPR, even the false edges are the same in each "split" because the false edge sets include ALL disconnected nodes within a certain number of hops.

Thus, for example, to perform the 10 AUC evaluations, a model would train 10 separate models on `train_edges.txt`, each validated by a different validation set (e.g. The first model would be validated on `validation_true_edges.txt` and `validation_false_edges_0.txt` - the second on `validation_true_edges.txt` and `validation_false_edges_1.txt`). Similarly, all models test with the same true edge set (`test_true_edges.txt` -- i.e. the last bucket) but 10 different false edge sets (e.g. `test_false_edges_0.txt`, `test_false_edges_1.txt`, ...).

Edge files have one edge per line, where edge (`u`, `v`, `time`, `weight`) is simply represented as `u`, `v`, `time`, and `weight` separated by space characters:
`u` `v` `time` `weight`
