#!/bin/bash

#SBATCH --partition=tc
#SBATCH -A mddlgp
#SBATCH -J DL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=240:00:00
#SBATCH --output=Result-%x.%j.out
#SBATCH --exclusive

nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

DIR_SRC=${SLURM_SUBMIT_DIR}
DIR_DATA="/beegfs/f2rw/Dataset_blended/jitter_150ms"
DIR_CONT="${DIR_SRC}/container"

LABEL="5D"
TRAIN_MODEL="true"

echo "Running training..."

srun singularity exec \
-B ${DIR_SRC}:/home/src \
-B ${DIR_DATA}:/home/data \
--nv ${DIR_CONT}/deepprecs.sif \
python /home/src/workflow.py \
--label=${LABEL} \
--train_model=${TRAIN_MODEL}