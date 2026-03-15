#!/bin/bash -eux
#SBATCH --job-name=jaguar-reid
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linus.loell@student.hpi.de
#SBATCH --partition=gpu-batch # -p
#SBATCH --cpus-per-task=3 # -c
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log # %j is job id


echo '=================== GPU ================'
source .venv/bin/activate

echo '=================== GPU ================'
nvidia-smi

echo '=================== Notebook  ================'
export RUN_CONFIG_PATH="configs/c2-dinov3-cosine.json"
jupyter nbconvert --to notebook --execute ./jaguar-ident.ipynb
