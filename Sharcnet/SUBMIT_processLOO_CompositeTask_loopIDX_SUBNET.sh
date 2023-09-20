#!/bin/bash

# Use this file to submit the individual batch submission files created for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/projects/def-beckers/shaws5/Research_code"
sh_file_prefix="processLOO_CompositeTask_loopIDX_"
loopIDX_start1=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)
loopIDX_end1=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)
for startIdx in {1..2} 
for Idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13
do
    startIDX=${loopIDX_start1[Idx]}
    endIDX=${loopIDX_end1[Idx]}
curr_file=processLOO_CompositeTask_loopIDX_${startIDX}to${endIDX}_SUBNET
	sbatch $base_path_rc/EEGnet/Sharcnet/sh_files/$curr_file.sh
	echo Submitted job $curr_file
	sleep 1 # Add a 1 second delay between each execution
done
echo Finished submitting jobs: All done
