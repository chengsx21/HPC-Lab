#!/bin/bash

# sh run.sh ./odd_even_sort 100 ./data/100.dat
# sh run.sh ./odd_even_sort 1000 ./data/1000.dat
# sh run.sh ./odd_even_sort 10000 ./data/10000.dat
# sh run.sh ./odd_even_sort 100000 ./data/100000.dat
# sh run.sh ./odd_even_sort 1000000 ./data/1000000.dat
# sh run.sh ./odd_even_sort 10000000 ./data/10000000.dat
# sh run.sh ./odd_even_sort 100000000 ./data/100000000.dat

# run on 2 machine * 48 process
srun -N 2 -n 48 $*

