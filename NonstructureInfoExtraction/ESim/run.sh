#!/bin/bash
node_file="../hin/node.dat"
link_file="../hin/link.dat"
path_file="../hin/path.dat"
output_file="../hin/vec.dat"

make

size=100 # embedding dimension
negative=5 # number of negative samples
samples=1 # number of edges (Million) for training at each iteration
iters=500 # number of iterations
threads=20 # number of threads for training

./bin/esim -model 2 -alpha 0.025 -node ${node_file} -link ${link_file}  -path ${path_file} -output ${output_file} -binary 1 -size ${size} -negative ${negative} -samples ${samples} -iters ${iters} -threads ${threads}