#!/bin/bash

# Use this file to submit the individual batch submission files created for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/projects/def-beckers/shaws5/Research_code"
sh_file_prefix="Cu_Ft_mRMR_group_Dp_CompositeTask_1_loopIDX_"
loopIDX_start1=(48 47 92 96 97 51 50 93 9 8 81 20 91 21 79 83 94 95 28 23 14 29 13 25 19 18 67 7 99 27 10 80 65 55 17 64 36 54 26 60 16 11 46 68 24 34 59 22 69)
loopIDX_end1=(48 47 92 96 97 51 50 93 9 8 81 20 91 21 79 83 94 95 28 23 14 29 13 25 19 18 67 7 99 27 10 80 65 55 17 64 36 54 26 60 16 11 46 68 24 34 59 22 69)
#for startIdx in {1..2} 
for Idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
do
    startIDX=${loopIDX_start1[Idx]}
    endIDX=${loopIDX_end1[Idx]}
curr_file=Cu_Ft_mRMR_group_Dp_CompositeTask_1_loopIDX_${startIDX}to${endIDX}_NET
	sbatch $base_path_rc/EEGnet/Sharcnet/sh_files/$curr_file.sh
	echo Submitted job $curr_file
	sleep 1 # Add a 1 second delay between each execution
done
echo Finished submitting jobs: All done
