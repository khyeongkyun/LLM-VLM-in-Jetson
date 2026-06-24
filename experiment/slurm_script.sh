#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J opt-pruning
#SBATCH --output=opt-pruning_%j.out
#SBATCH --error=opt-pruning_%j.err
#SBATCH --time=2-00:00:00
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/LLM-VLM-in-Jetson/experiment/llm-streamline/LLM-Streamline-main

# ======== General resourece setting
#SBATCH --mem=512gb
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --cpus-per-task=4

# ======== GPU resource setting
#SBATCH --gres=gpu:1

# load required modules
module load slurm_setup
eval "$(micromamba shell hook --shell bash)"
micromamba activate llm-streamline
export PYTHONNOUSERSITE=1
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export HYDRA_FULL_ERROR=1
export HF_HOME=/dss/dsstbyfs02/scratch/07/di54rur/.huggingface


cd /dss/dsshome1/07/di54rur/kim_he/LLM-VLM-in-Jetson/experiment/llm-streamline/LLM-Streamline-main

python mseloss_entry.py \
  --model_name facebook/opt-6.7b \
  --layer_intervals 8 \
  --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned