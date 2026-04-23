#!/bin/bash

#SBATCH --partition=tc
#SBATCH -A mddlgp
#SBATCH -J DL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=100:00:00
#SBATCH --output=./slurms/Result-%x.%j.out
#SBATCH --error=./slurms/Result-%x.%j.err
#SBATCH --exclusive

nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

DIR_SRC=${SLURM_SUBMIT_DIR}
DIR_DATA="/beegfs/gaia/tcs/ufrj_ml_u30s/dados/Deblending"
DIR_CONT="/beegfs/g86s/container"

LABEL="marmousi_norm_abs"
TRAIN_MODEL="true"

echo "Running training..."

srun singularity exec \
-B ${DIR_SRC}:/home/src \
-B ${DIR_DATA}:/home/data \
--nv ${DIR_CONT}/deepprecs.sif \
python /home/src/workflow.py \
--label=${LABEL} \
--train_model=${TRAIN_MODEL}