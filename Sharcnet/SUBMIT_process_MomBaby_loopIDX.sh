#!/bin/bash

# Use this file to submit the individual batch submission files created for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/projects/def-beckers/shaws5/Research_code"
sh_file_prefix="process_MomBaby_loopIDX_"
loopIDX_step=2

for startIdx in 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97 99 101 103 105 107 109 111 113 115 117 119 121 123 125 127 129 131 133 135 137 139 141 143 145 147 149 151 153 155 157 159 161 163 165 167 169 171 173 175 177 179 181 183 185 187 189 191 193 195 197 199 201 203 205 207 209 211 213 215 217 219 221 223 225 227 229 231 233 235 237 239 241 243 245 247 249 251 253 255 257 259 261 263 265 267 269 271 273 275
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
