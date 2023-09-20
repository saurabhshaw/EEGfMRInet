#!/bin/bash
#SBATCH --cpus-per-task=24   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-beckers

module load matlab
cd
matlab -nodesktop -nosplash -nodisplay -r "run('main_submit.m'); exit"