#! /bin/bash
#SBATCH --job-name="ce_masked"
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0:0

# Executable
srun /bin/bash run.sh