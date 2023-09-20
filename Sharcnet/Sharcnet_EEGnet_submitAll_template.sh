#!/bin/bash

# Use this file to submit the individual batch submission files created for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/Research_code"
sh_file_prefix="process_RunningNFB_loopIDX_"
loopIDX_step=2

for startIdx in {1..2} 
do	
	# chmod +x /home/shaws5/Group_analysis_submit_bw_Sub_sbatch_Run$run.sh
	# ./Group_analysis_submit_bw_Sub_sbatch_Run$run.sh
	endIdx=$(($startIdx+$loopIDX_step-1))
	curr_file=$sh_file_prefix${startIdx}to${endIdx}
	sbatch $base_path_rc/EEGnet/Sharcnet/sh_files/$curr_file.sh
	echo Submitted job $curr_file
	sleep 1 # Add a 1 second delay between each execution
done
echo Finished submitting jobs: All done