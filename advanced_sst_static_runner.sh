# Usage: nice -n <nice number> ./advanced_sst_static_runner.sh <graph name> <num_processors> <directed>
#
# <directed> can be --directed or simply left blank for undirected

python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 0
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 1
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 2
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 3
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 4
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 5
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 6
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 7
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 8
python runner.py --model SST_SVM --input $1 --num_proc $2 --partial count $3 --idx 9

python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 0 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 1 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 2 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 3 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 4 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 5 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 6 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 7 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 8 &
python runner.py --model SST_SVM --input $1 --partial fit $3 --idx 9

python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 0
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 1
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 2
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 3
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 4
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 5
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 6
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 7
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 8
python runner.py --model SST_SVM --input $1 --partial "eval" $3 --idx 9
