#!/bin/bash

# Use this file to submit the individual batch submission files created for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/projects/def-beckers/shaws5/Research_code"
sh_file_prefix="process_RunningNFB_loopIDX_"
loopIDX_step=1

for startIdx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144
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
