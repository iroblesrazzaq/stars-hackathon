#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

echo "output of the visible GPU environment"
nvidia-smi

module load python/miniforge-24.1.2 # python 3.10

# Use hackathon environment
source /project/dfreedman/hackathon/ismael-mason-darren-vincent/ismael/new_starsenv/bin/activate

echo Running XGBoost script
python xgboost_script.py /project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/compas-data.pkl
