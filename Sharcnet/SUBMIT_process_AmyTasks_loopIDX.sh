#!/bin/bash

# Use this file to submit the individual batch submission files created for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/projects/def-beckers/shaws5/Research_code"
sh_file_prefix="process_AmyTasks_loopIDX_"
loopIDX_start1=(2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13)
loopIDX_end1=(2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13)
loopIDX_start2=(1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3)
loopIDX_end2=(1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3)
	# chmod +x /home/shaws5/Group_analysis_submit_bw_Sub_sbatch_Run$run.sh
for Idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
do
    startIDX=${loopIDX_start1[Idx]}_${loopIDX_start2[Idx]}
    endIDX=${loopIDX_end1[Idx]}_${loopIDX_end2[Idx]}
curr_file=process_AmyTasks_loopIDX_${startIDX}to${endIDX}
	sbatch $base_path_rc/EEGnet/Sharcnet/sh_files/$curr_file.sh
	echo Submitted job $curr_file
	sleep 1 # Add a 1 second delay between each execution
done
echo Finished submitting jobs: All done
