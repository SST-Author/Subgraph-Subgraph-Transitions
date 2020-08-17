# Test Results

These directories contain results from AUC and AUPR scores for link prediction tasks. (Results produced by runner.py)

The directory structure mirrors that of the `evaluation_data_splits/` folder. See the Readme in `evaluation_data_splits/` for more information on the directory structure.

## File Names

Result files begin with the model name, followed by either "auc" or "aupr\_at\_k\<k value\>", and then finished with "\_\<idx\>.pkl".

For example, `SST_SVM_aupr_at_k2_3.pkl`.

Each index corresponds to a different train/validation/test split for the graph in question. To understand the k-values for the AUPR files, see the Readme in `evaluation\_data\_splits/`.

## File Contents

An explanation of file contents can perhaps be best explained with an example python shell run:

```
python
Python 3.7.8 | packaged by conda-forge | (default, Jul 31 2020, 02:25:08)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pickle as pkl
>>> f = open("test_results/static/karate_undirected/SST_SVM_auc_0.pkl", "rb")
>>> auc_values_dict = pkl.load(f)
>>> f.close()
>>> [key for key, value in auc_values_dict.items()]
['auc', 'fprs', 'tprs']
>>> auc_values_dict['auc']
0.8163265306122449
>>> f = open("test_results/static/karate_undirected/SST_SVM_aupr_at_k2_0.pkl", "rb")
>>> aupr_values_dict = pkl.load(f)
>>> [key for key, value in aupr_values_dict.items()]
['aupr', 'generous_aupr', 'stingy_aupr', 'recalls', 'precisions']
>>> aupr_values_dict['aupr']
0.23185312228117738
>>> f.close()
>>> f = open("test_results/static/karate_undirected/SST_SVM_ssts_0.pkl", "rb")
>>> sorted_weighted_ssts = pkl.load(f)
>>> sorted_weighted_ssts[0]
(0.6140689813237079, ('Node Highlights:', (0, 0, None, None), 'Node Traits:', ('InvNode Degree:', ('Equal', 'Equal', None, None)), 'Edge List:', ((0, 1), (0, 3), (1, 2), (1, 3)), 'Edge Traits:', ()))
>>> sorted_weighted_ssts[1]
(0.4599728248367558, ('Node Highlights:', (0, 0, None, None), 'Node Traits:', ('InvNode Degree:', ('Equal', 'Equal', None, None)), 'Edge List:', ((0, 1), (0, 3), (1, 2), (2, 3)), 'Edge Traits:', ()))
>>> sorted_weighted_ssts[2]
(-0.3812446404369189, ('Node Highlights:', (0, 0, None, None), 'Node Traits:', ('InvNode Degree:', ('Equal', 'Equal', None, None)), 'Edge List:', ((0, 1), (0, 3), (1, 2)), 'Edge Traits:', ()))
>>> sorted_weighted_ssts[3]
(-0.3069621851089166, ('Node Highlights:', (0, 0, None, None), 'Node Traits:', ('InvNode Degree:', ('Equal', 'Equal', None, None)), 'Edge List:', ((0, 1), (1, 2), (1, 3)), 'Edge Traits:', ()))
>>> len(sorted_weighted_ssts)
25
>>> f.close()
>>> exit()
```

### For AUC:

`auc` -- the area under the ROC curve

`tprs` -- a list -- the true positive rates (the y axis of the ROC curve)

`fprs` -- a list -- the false positive rates (the x axis of the ROC curve)

### For AUPR:

`aupr` -- the area under the precision-recall curve, calculated with the trapezoidal rule

`generous_aupr` -- for debugging -- denotes a calculation of aupr where the max of two y coordinates is used rather than the trapezoid rule

`stingy_aupr` -- for debugging -- denotes a calculation of aupr where the min of two y coordinates is used rather than the trapezoid rule

`precisions` -- a list -- the precisions (the y axis of the AUPR curve, listed right to left)

`recalls` -- a list -- the recalls (the x axis of the AUPR curve, listed right to left)

### For SSTS:

The pickle file will contain a list of (weight, subgraph-subgraph transition) pairs in order of decreasing magnitude of weight.

The "weight" is a component of the (unit) direction vector from a linear SVM separating non-changes from changes. Thus, the larger a weight, the more that particular SST indicates a change, and smaller (i.e. a "larger" negative) a weight the more it indicates a non-change.

An SST is represented as a four-tuple: `(Node Highlights, Node Traits, Edge List, Edge Traits)`

 - Node Highlights indicates which nodes are the ones involved in the change.
    - For node additions, the new node has a highlight of `0` and other nodes have `None`.
    - For edge additions, the source node has a hightlight of `0` and the target has `0` if undirected and `1` if directed. Other nodes have `None`
    - Node deletions and edge deletions correspond to their respective additions.
- Node Traits indicates any trait values nodes have. For more information on node traits, check out the documentation in `graph_change_feature_counts.py`
    - In the link-prediction experiments performed in this repository, there is one node trait: `InvNode Degree`
        - InvNode Degree is only used in static, undirected link prediction.
        - It states which of the two nodes being connected, if either, has a higher degree.
- Edge List has a list of the edges in the subgraph.
    - If edges for a node or edge addition, includes all the edges after the SST is completed
    - If edges for a node or edge deletion, includes all the edges before the SST is completed
- Edge Traits -- same as Node Traits but for edges
    - In the link-prediction experiments performed in this repository, there are two edge traits: `TLP: Freq` and `TLP: Recency` (`temporal_link_pred_traits.py`)
        - `TLP: Freq` indicates how many timestamps an edge (i.e. interaction) has occurred before it occurred in the current SST
            - Possible values are "0" (this is the first time this interaction occurred), "1", "2", "3+" (indicating 3 or more prior occurrences)
            - *After* the SST is completed (i.e. after the edge is added), the new edge will update its value (e.g. from "0" to "1" or "3+" to "3+").
        - `TLP: Recency` indicates how recently an edge (i.e. interaction) occured before it occurred in the current SST
            - Possible values are "Never" (this is the first time this interaction occurred), "Newest" (it occured in the previous timestamp), "New" (it occurred in the timestamp before the last), and "Old" (Latest prior occurrence was at least 3 timestamps ago)
            - *After* the SST is completed (i.e. after the edge is added), the new edge will update its value to "Newest", and any edges not added by some SST will "age" (e.g. "Newest" -> "New", "Old" -> "Old")
