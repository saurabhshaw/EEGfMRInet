#!/bin/bash

# Use this file to create individual batch submission files for job submission:
# Make sure to make this file executable using chmod and executing in the command line as follows:
# chmod +x /home/shaws5/CLSA_NN_latest_splitSubmit.sh
# ./CLSA_NN_latest_splitSubmit.sh

base_path_rc="/home/shaws5/Research_code"
base_path_rd="/scratch/shaws5/Research_data"
template_input_file="Sharcnet_main_process_JFL_Wobbie.m"
function_description='Feature_computation'
req_cores=22 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
req_RAM=128 # memory per node
req_queue='threaded'
req_runtime='0-01:00' # time (DD-HH:MM)
num_epochs=146
split_size=10


module load matlab

for sesIdx in {1..2} 
do
	for ((epoch=1;epoch<=${num_epochs};epoch=$(($epoch+$split_size))));
  do		
		#input_file="${base_path_rc}/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.m"
		#input_filename="Sharcnet_main_process_JFL_Subject${subIdx}_Session${sesIdx}_Epoch${epoch}"
		#output_file="${base_path_rc}/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/out_files/${input_filename}_run_${req_cores}_${req_RAM}G.log"
		#job_description="${function_description}_${req_cores}_${req_RAM}"
		#run_command="sqsub -q ${req_queue} -n ${req_cores} --mpp ${req_RAM}G -r ${req_runtime} -o ${output_file} -i ${input_file} -j ${job_description} matlab"

		
		
		# sed "545s#.*#k=$run#" /home/shaws5/Research_code/EEG_fMRI_Modelling/Deep_Learning/Group_analysis_bw_Sub_split.py > /home/shaws5/Research_code/EEG_fMRI_Modelling/Deep_Learning/Sharcnet/Group_analysis_bw_Sub_split_Run$run.py
   
   start_idx=$epoch
   end_idx=$(($epoch+$split_size-1))
   if ((${end_idx}>${num_epochs}))
   then
     end_idx=${num_epochs}
   fi
   
   sed "2s#.*#\#SBATCH --cpus-per-task=${req_cores}#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/Sharcnet_main_process_JFL_submit.sh > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfileSubject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh
   sed "3s#.*#\#SBATCH --mem=${req_RAM}G#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfileSubject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile2Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh
   sed "4s#.*#\#SBATCH --time=${req_runtime}#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile2Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile3Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh
		sed "5s#.*#\#SBATCH --output=${base_path_rc}/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/out_files/Node%N-JobID%j-Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.out#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile3Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile4Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh
    sed "8s#.*#cd $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile4Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile5Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh 
		sed "9s#.*#matlab -nodesktop -nosplash -nodisplay -r \"run('Sharcnet_main_process_JFL_Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.m'); exit\"#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile5Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_fileSubject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh 
		sed "2s#.*#subIdx = $subIdx;#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/$template_input_file > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp1.m 
		sed "3s#.*#sesIdx = $sesIdx;#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp1.m > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp2.m 
		sed "4s#.*#epoch_range = [${start_idx}:${end_idx}];#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp2.m > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp3.m 
		sed "5s#.*#numCores = $req_cores;#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp3.m > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp4.m 
		sed "6s#.*#base_path_rc = '${base_path_rc}';#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp4.m > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp5.m 
		sed "7s#.*#base_path_rd = '${base_path_rd}';#" $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp5.m > $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_Subject${subIdx}_Session${sesIdx}_Epoch${epoch}.m

		# chmod +x /home/shaws5/Group_analysis_submit_bw_Sub_sbatch_Run$run.sh
		# ./Group_analysis_submit_bw_Sub_sbatch_Run$run.sh
		sbatch $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_fileSubject${subIdx}_Session${sesIdx}_Epoch${epoch}.sh
		echo Submitted job for Subject $subIdx Session $sesIdx Epoch $epoch from $start_idx to $end_idx
		sleep 1 # Add a 5 second delay between each execution
	done
done
#done

echo Finished submitting jobs, deleting created files
rm -r $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_tempfile*
# rm -r $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sh_files/Sharcnet_main_process_JFL_submit_file*
rm -r $base_path_rc/EEG_fMRI_Modelling/Joint_Feature_Learning/Sharcnet/sub_files/Sharcnet_main_process_JFL_temp*
echo All done


# python /home/shaws5/Research_code/EEG_fMRI_Modelling/Deep_Learning/Group_analysis_bw_Sub.py