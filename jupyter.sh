#!/bin/bash -eux
#SBATCH --job-name=jupyter
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --output=jupyter_%j.out
#SBATCH --partition=cpu-batch # -p
#SBATCH --cpus-per-task=1 # -c
#SBATCH --mem=8gb
#SBATCH --time=08:00:00

# uncomment to receive email when job fails
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=firstname.lastname@student.hpi.de

# Initialize conda:
eval "$(conda shell.bash hook)"

# Set the correct IP variable:
IP=$(hostname -I | tr ' ' '\n' | grep -m1 '^10\.130\.')

# some command to activate a specific conda environment or whatever:
# ...
source .venv/bin/activate

jupyter-lab --ip $IP --port=60666
