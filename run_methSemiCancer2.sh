#!/bin/bash

source_X_filename="./example_data/example_source_X.csv"
source_y_filename="./example_data/example_source_y.csv"
target_X_filename="./example_data/example_target_X.csv"
result_dir="./results"

mkdir -p $result_dir

python3 methSemiCancer2.py $source_X_filename $source_y_filename $target_X_filename $result_dir
