#!/bin/sh
#BSUB -q gpuv100
#BSUB -J test
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "select[gpu32gb]"

#BSUB -m "n-62-11-13"


nodes=$LSB_HOSTS
nodes_array=($nodes)
head_node=${nodes_array[0]}
this_node=$HOSTNAME
if [ "$head_node" = "$this_node" ]
then
    head_node_ip="$(hostname --ip-address)"
fi
echo $nodes
echo $head_node
echo $head_node_ip

nvidia-smi