# Subgraph-Subgraph Transitions ("SSTs")

Corresponds to the paper: "Joint Subgraph-to-Subgraph Transitions -- Generalizing Triadic Closure for Interpretable Graph Modeling" at (Waiting for Paper Acceptance to add Link).

![Example of SSTs](https://github.com/SST-Author/Subgraph-Subgraph-Transitions/blob/master/images/first_paper_figure.png?raw=true)

Figure copied from the aforementioned paper.

## This Repository Contains...

 - Code to model a change to a graph in terms of how that change causes one or more transitions of subgraphs to different subgraphs.
 - Code to perform link prediction tasks using these features coupled with an extremely simple model (linear SVM).
 - Code to compare this simple link prediction method against some state-of-the-art link prediction methods (as of August 2020).
 - Code to produce human-interpretable understanding of the SVM's decision.
 - Evaluation results/output.

## Dependencies for the Software

We use a single `conda` environment for the entire project, including running the neural nets. To install the dependencies with `conda`, perform the following:

First, go to `envs/` and run `conda env create -f neuralnets.yml`.
As a part of creating this environment, TensorFlow 1.15.0, PyTorch 1.6.0, and TorchVision 0.7.0 should be installed.

You are now ready to run all the code in this repository. If you do not have `conda` installed, you can install it from here: `https://docs.conda.io/en/latest/miniconda.html`.

## If You Only Care about Subgraph-to-Subgraph Transition Counting...

Then check out the following files, as they will be of the most use to you:
 - `graph_change_feature_counts.py`
 - `subgraph_change_labeler.py`

For an example of using these files, check out `sst_svm_modeler.py`.

Secondarily, consider looking at the files:
 - `trait_updater.py`
    - `temporal_link_pred_traits.py`
    - `degree_trait.py`
 - `graph_data.py`
 - `graph_change.py`

## Datasets and Preprocessing

Our datasets are in the `datasets/` folder, where you can find another Readme explaining the graphs and format.

To convert these datasets into train/validation/test splits for AUC and AUPR evaluations, use `evaluation_data_splits.py`.

For example, to create 10 train/validation/test splits of the karate graph modeled as a static undirected graph, with test sets for evaluating AUPR on the link-prediction task of disconnected nodes within 3 hops (see Readme in the `evaluation_data_splits/` folder for more details on the AUPR), we would run:

`python evaluation_data_splits.py datasets/karate.g static undirected aupr_k=3 10 overwrite`

See the Readme in the `evaluation_data_splits` folder for more details on the output of this command and how to interpret the train/validation/test split files.

## To Create Small Train/Validation/Test Splits to Test On

Use this for some quick-and-dirty testing to make sure everything is operable. Takes the famous karate graph and treats it as four different kinds of graphs, static-undirected, static-directed, temporal-undirected, temporal-directed.

```
python evaluation_data_splits.py datasets/karate.g static undirected aupr_k=2:4 10 overwrite
python evaluation_data_splits.py datasets/karate.g static directed aupr_k=2:4 10 overwrite
python evaluation_data_splits.py datasets/karate-temporal.g temporal undirected aupr_k=2:4 10 overwrite
python evaluation_data_splits.py datasets/karate-temporal.g temporal directed aupr_k=2:4 10 overwrite
```

## To Create all the Splits Used in the Paper

(Note that there is some randomness, so the splits will not be identical.)

```
python evaluation_data_splits.py datasets/cora.g static undirected aupr_k=2:4 10 overwrite
python evaluation_data_splits.py datasets/cora.g static directed aupr_k=2:4 10 overwrite

python evaluation_data_splits.py datasets/citeseer.g static undirected aupr_k=2:4 10 overwrite
python evaluation_data_splits.py datasets/citeseer.g static directed aupr_k=2:4 10 overwrite

python evaluation_data_splits.py datasets/eucore.g static undirected aupr_k=2:4 10 overwrite
python evaluation_data_splits.py datasets/eucore.g static directed aupr_k=2:4 10 overwrite

python evaluation_data_splits.py datasets/college-temporal.g temporal directed aupr_k=2:4 10 overwrite

python evaluation_data_splits.py datasets/eucore-temporal.g temporal directed aupr_k=2:4 10 overwrite

python evaluation_data_splits.py datasets/wiki-en-additions.g temporal undirected no_aupr 10 overwrite

python evaluation_data_splits.py datasets/synthetic/BA_2_1000_raw_dir.g temporal directed no_aupr 1 overwrite
```

## Running Link Prediction

For a quick and simple run, try creating the karate data splits; then run the following command:

`python runner.py --model SST_SVM --input karate --idx 0 --num_proc 2`

## To Create the Results in the Paper

(Note that there is some randomness, so the results will not be identical, but should be similar.)

### Static Link Prediction Tests

To run the 4-node SST SVMs on all 10 train/validation/test splits for the graphs using up to 30 processes:

```
./advanced_sst_static_runner.sh cora 30
./advanced_sst_static_runner.sh cora 30 --directed
./advanced_sst_static_runner.sh citeseer 30
./advanced_sst_static_runner.sh citeseer 30 --directed
./advanced_sst_static_runner.sh eucore 30
./advanced_sst_static_runner.sh eucore 30 --directed

./advanced_sst_temporal_runner.sh college-temporal 30 --directed
./advanced_sst_temporal_runner.sh eucore-temporal 30 --directed
./advanced_sst_temporal_runner.sh wiki-en-additions 30 --directed
```

To run the 3-node SST STMs, open `runner.py` and replace `subgraph_size = 4` with `subgraph_size = 3` in the `other_runner()` function, then repeat the above commands.

To run the static undirected neural networks, run the following. Note that `--not_parallel` does not force the neural network code to run sequentially, rather, `--not_parallel` has the neural network model run on one of the 10 train/validation/test splits at a time.

```
./run_model_on_static_graphs.sh Linear_AE --not_parallel
./run_model_on_static_graphs.sh Linear_VAE --not_parallel
./run_model_on_static_graphs.sh GCN_AE --not_parallel
./run_model_on_static_graphs.sh GCN_VAE --not_parallel
```

Static directed neural networks:
```
./run_model_on_static_graphs.sh Gravity_GCN_AE --not_parallel --directed
./run_model_on_static_graphs.sh Gravity_GCN_VAE --not_parallel --directed
```

Random and CommonNeighbors models:
```
./run_model_on_static_graphs.sh Random --not_parallel
./run_model_on_static_graphs.sh Random --not_parallel --directed
./run_model_on_static_graphs.sh CommonNeighbors --not_parallel
./run_model_on_static_graphs.sh CommonNeighbors --not_parallel --directed
```

### Temporal Link Prediction Tests

4-Node SST SVM using up to 30 processes (you can vary the number from 1 to whatever).

```
./advanced_sst_temporal_runner.sh college-temporal 30 --directed
./advanced_sst_temporal_runner.sh eucore-temporal 30 --directed
./advanced_sst_temporal_runner.sh wiki-en-additions 30 --directed
```

To run the 3-node SST STMs, open `runner.py` and replace `subgraph_size = 4` with `subgraph_size = 3` in the `other_runner()` function, then repeat the above commands.

To run the graph neural network:

`./run_model_on_temporal_graphs.sh TGN --not_parallel`

To run the random baseline:

`./run_model_on_temporal_graphs.sh Random --not_parallel`

### Barabasi Albert SSTs

Use the following command for 4-node SSTs:

`python runner.py --model SST_SVM --input BA_2_1000_raw_dir --directed --temporal --idx 0`

To run the 3-node SST STMs, open `runner.py` and replace `subgraph_size = 4` with `subgraph_size = 3` in the `other_runner()` function, then repeat the above command.

## Inspecting Test Results

To see information on loading test results, check out the Readme in the `test_results` folder.

## Thanks and Citations

TODO: Add citations to the graphs in our datasets.
