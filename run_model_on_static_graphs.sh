# Usage:
# ./run_model_on_static_graphs.sh <model> <parallel> <directed>
#
# <model> -- a model name, such as Linear_AE or SST_SVM
# <parallel> -- input --parallel to get all 10 splits of a graph to run in
#    parallel. input anything else to get them to run in sequence
# <directed> -- if you want to run on the directed version of a graph, input
#    --directed otherwise leave blank.

if [ "$2" = "--parallel" ];
then
    python runner.py --model $1 $3 --input karate --idx 0 &
    python runner.py --model $1 $3 --input karate --idx 1 &
    python runner.py --model $1 $3 --input karate --idx 2 &
    python runner.py --model $1 $3 --input karate --idx 3 &
    python runner.py --model $1 $3 --input karate --idx 4 &
    python runner.py --model $1 $3 --input karate --idx 5 &
    python runner.py --model $1 $3 --input karate --idx 6 &
    python runner.py --model $1 $3 --input karate --idx 7 &
    python runner.py --model $1 $3 --input karate --idx 8 &
    python runner.py --model $1 $3 --input karate --idx 9

    python runner.py --model $1 $3 --input cora --idx 0 &
    python runner.py --model $1 $3 --input cora --idx 1 &
    python runner.py --model $1 $3 --input cora --idx 2 &
    python runner.py --model $1 $3 --input cora --idx 3 &
    python runner.py --model $1 $3 --input cora --idx 4 &
    python runner.py --model $1 $3 --input cora --idx 5 &
    python runner.py --model $1 $3 --input cora --idx 6 &
    python runner.py --model $1 $3 --input cora --idx 7 &
    python runner.py --model $1 $3 --input cora --idx 8 &
    python runner.py --model $1 $3 --input cora --idx 9

    python runner.py --model $1 $3 --input citeseer --idx 0 &
    python runner.py --model $1 $3 --input citeseer --idx 1 &
    python runner.py --model $1 $3 --input citeseer --idx 2 &
    python runner.py --model $1 $3 --input citeseer --idx 3 &
    python runner.py --model $1 $3 --input citeseer --idx 4 &
    python runner.py --model $1 $3 --input citeseer --idx 5 &
    python runner.py --model $1 $3 --input citeseer --idx 6 &
    python runner.py --model $1 $3 --input citeseer --idx 7 &
    python runner.py --model $1 $3 --input citeseer --idx 8 &
    python runner.py --model $1 $3 --input citeseer --idx 9

    python runner.py --model $1 $3 --input eucore --idx 0 &
    python runner.py --model $1 $3 --input eucore --idx 1 &
    python runner.py --model $1 $3 --input eucore --idx 2 &
    python runner.py --model $1 $3 --input eucore --idx 3 &
    python runner.py --model $1 $3 --input eucore --idx 4 &
    python runner.py --model $1 $3 --input eucore --idx 5 &
    python runner.py --model $1 $3 --input eucore --idx 6 &
    python runner.py --model $1 $3 --input eucore --idx 7 &
    python runner.py --model $1 $3 --input eucore --idx 8 &
    python runner.py --model $1 $3 --input eucore --idx 9
else
    python runner.py --model $1 $3 --input karate --idx 0
    python runner.py --model $1 $3 --input karate --idx 1
    python runner.py --model $1 $3 --input karate --idx 2
    python runner.py --model $1 $3 --input karate --idx 3
    python runner.py --model $1 $3 --input karate --idx 4
    python runner.py --model $1 $3 --input karate --idx 5
    python runner.py --model $1 $3 --input karate --idx 6
    python runner.py --model $1 $3 --input karate --idx 7
    python runner.py --model $1 $3 --input karate --idx 8
    python runner.py --model $1 $3 --input karate --idx 9

    python runner.py --model $1 $3 --input cora --idx 0
    python runner.py --model $1 $3 --input cora --idx 1
    python runner.py --model $1 $3 --input cora --idx 2
    python runner.py --model $1 $3 --input cora --idx 3
    python runner.py --model $1 $3 --input cora --idx 4
    python runner.py --model $1 $3 --input cora --idx 5
    python runner.py --model $1 $3 --input cora --idx 6
    python runner.py --model $1 $3 --input cora --idx 7
    python runner.py --model $1 $3 --input cora --idx 8
    python runner.py --model $1 $3 --input cora --idx 9

    python runner.py --model $1 $3 --input citeseer --idx 0
    python runner.py --model $1 $3 --input citeseer --idx 1
    python runner.py --model $1 $3 --input citeseer --idx 2
    python runner.py --model $1 $3 --input citeseer --idx 3
    python runner.py --model $1 $3 --input citeseer --idx 4
    python runner.py --model $1 $3 --input citeseer --idx 5
    python runner.py --model $1 $3 --input citeseer --idx 6
    python runner.py --model $1 $3 --input citeseer --idx 7
    python runner.py --model $1 $3 --input citeseer --idx 8
    python runner.py --model $1 $3 --input citeseer --idx 9

    python runner.py --model $1 $3 --input eucore --idx 0
    python runner.py --model $1 $3 --input eucore --idx 1
    python runner.py --model $1 $3 --input eucore --idx 2
    python runner.py --model $1 $3 --input eucore --idx 3
    python runner.py --model $1 $3 --input eucore --idx 4
    python runner.py --model $1 $3 --input eucore --idx 5
    python runner.py --model $1 $3 --input eucore --idx 6
    python runner.py --model $1 $3 --input eucore --idx 7
    python runner.py --model $1 $3 --input eucore --idx 8
    python runner.py --model $1 $3 --input eucore --idx 9

fi
