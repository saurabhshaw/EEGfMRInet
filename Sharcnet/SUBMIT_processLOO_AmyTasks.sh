#!/bin/bash
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G        # memory per node
#SBATCH --time=0-11:50      # time (DD-HH:MM)
#SBATCH --output=/home/shaws5/projects/def-beckers/shaws5/Research_code/EEGnet/Sharcnet/out_files/processLOO_AmyTasks-Node%N-JobID%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-beckers

module load matlab
cd /home/shaws5/projects/def-beckers/shaws5/Research_code/EEGnet/Main
matlab -nodesktop -nosplash -nodisplay -r "run('processLOO_AmyTasks.m'); exit"
