#!/bin/bash
#BSUB -J mnode_torchrun
### -- Select the resources: 
### --- number of total processes
#BSUB -n 1
### --- number of procs/node
#BSUB -R "span[ptile=1]"
### --- number of GPUs per node (must match the ptile number!)
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
### --- other resources per process (memory and CPU cores)
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "affinity[core(4)]"
### --- the GPU queue
#BSUB -q gpuv100
### --- specify the wall clock time 
#BSUB -W 24:00
### -- My email address --
##BSUB -u s212711@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o top-view-mapping/out/%J.out
#BSUB -e top-view-mapping/out/%J.err

#BSUB -R "select[gpu32gb]"
# #BSUB -R "select[sxm2]"

# the master is the first node in the pool, executing this script
export MASTER_ADDR=$HOSTNAME

# get a free port for communication at runtime (better than a fixed
# port, which could be taken by some other process)
export FREEPORT=`python3 -c 'import socketserver; print(socketserver.TCPServer(("localhost", 0),None).server_address[1])'`

# get the list of hosts (for blaunch)
LHOSTS=$(awk ' {print $1}' $LSB_AFFINITY_HOSTFILE | uniq | paste -sd ' ')

# get the parameters for the torchrun call
NNODES=$(awk ' {print $1}' $LSB_AFFINITY_HOSTFILE | uniq | wc -l)
PTILE=$(echo $LSB_MCPU_HOSTS | awk '{print $2}')

# uncomment for more information at startup
echo "running on ---$LHOSTS---"
echo "MASTER_ADDR="$MASTER_ADDR
echo "PORT="$FREEPORT
echo "NNODES="$NNODES
echo "PTILE="$PTILE

# uncomment the next line, to avoid I/O buffering in the python processes
export PYTHONUNBUFFERED=1

cd /work3/s212711
module load cuda/12.1.1
source env/bin/activate
cd top-view-mapping

blaunch -z "$LHOSTS" torchrun \
	--nnodes=$NNODES \
	--nproc_per_node=$PTILE \
	--rdzv_id=${LSB_JOBID} \
	--rdzv_backend=c10d \
	--rdzv_endpoint=${MASTER_ADDR}:${FREEPORT} \
    TRAIN.py apple