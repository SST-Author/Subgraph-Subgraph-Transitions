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
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 0 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 1 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 2 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 3 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 4 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 5 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 6 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 7 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 8 &
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 9

    python runner.py --model $1 $3 --temporal --input college-temporal --idx 0 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 1 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 2 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 3 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 4 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 5 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 6 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 7 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 8 &
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 9

    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 0 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 1 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 2 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 3 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 4 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 5 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 6 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 7 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 8 &
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 9

    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 0 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 1 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 2 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 3 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 4 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 5 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 6 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 7 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 8 &
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 9
else
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 0
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 1
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 2
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 3
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 4
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 5
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 6
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 7
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 8
    python runner.py --model $1 $3 --temporal --input karate-temporal --idx 9

    python runner.py --model $1 $3 --temporal --input college-temporal --idx 0
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 1
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 2
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 3
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 4
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 5
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 6
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 7
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 8
    python runner.py --model $1 $3 --temporal --input college-temporal --idx 9

    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 0
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 1
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 2
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 3
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 4
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 5
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 6
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 7
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 8
    python runner.py --model $1 $3 --temporal --input eucore-temporal --idx 9

    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 0
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 1
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 2
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 3
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 4
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 5
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 6
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 7
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 8
    python runner.py --model $1 $3 --temporal --input wiki-en-additions --idx 9

fi
