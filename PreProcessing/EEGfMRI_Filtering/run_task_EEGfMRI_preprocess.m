function [EEG] = run_task_EEGfMRI_preprocess(EEG,task_dir,scan_parameters,dataset_name,block_type,offline_preprocess_cfg,overwrite_files,base_path,m)

%% Verify the position of the ECG channel
% [EEG] = EEGfMRI_verify_ECGchannel(EEG,scan_parameters); 

% Check for more triggers than expected!!!!! - stray slice

%% Remove Gradient and BCG Artifacts:
[EEGfMRI_preprocess_param] = EEGfMRI_preprocess_setup();

if overwrite_files || isempty(dir([task_dir filesep 'EEGfMRI_PreProcessed' filesep 'task_' dataset_name '_' block_type '_TaskBlock' num2str(m) '_EEGfMRIpreprocessed.set']))
    temp_scan_parameters = scan_parameters; curr_num_images = length(find(strcmp(scan_parameters.slice_marker,{EEG.event(:).type})));
    if curr_num_images < scan_parameters.tfunc_num_images temp_scan_parameters.tfunc_num_images = curr_num_images; end
    % [EEG] = EEGfMRI_preprocess(EEG,temp_scan_parameters,task_dir,EEGfMRI_preprocess_param,overwrite_files);
    EEGfMRI_preprocess_compiled(EEG,temp_scan_parameters,task_dir,EEGfMRI_preprocess_param,overwrite_files,task_dir,base_path);
    EEG = pop_loadset('filename',['task_' dataset_name '_' block_type '_TaskBlock' num2str(m) '_EEGfMRIpreprocessed.set'],'filepath',[task_dir filesep 'EEGfMRI_PreProcessed']); EEG = eeg_checkset( EEG );
else
    EEG = pop_loadset('filename',['task_' dataset_name '_' block_type '_TaskBlock' num2str(m) '_EEGfMRIpreprocessed.set'],'filepath',[task_dir filesep 'EEGfMRI_PreProcessed']); EEG = eeg_checkset( EEG );
end

%% Truncate the start of the EEG to the specified number of seconds before the onset of the first MRI marker:
slice_latencies = [EEG.event(find(strcmp(scan_parameters.slice_marker,{EEG.event.type}))).latency]; first_slice_latency = slice_latencies(1);
if (first_slice_latency > (EEG.srate*offline_preprocess_cfg.preMRItruncation))
    truncate_latency = (first_slice_latency/EEG.srate) - offline_preprocess_cfg.preMRItruncation;
    EEG = pop_select(EEG, 'notime',[0 double(truncate_latency)]);
end

% Truncate the end of the EEG to the specified number of seconds after the last MRI marker:
slice_latencies = [EEG.event(find(strcmp(scan_parameters.slice_marker,{EEG.event.type}))).latency]; last_slice_latency = slice_latencies(length(slice_latencies));
truncate_latency_slices = (last_slice_latency + (EEG.srate*offline_preprocess_cfg.preMRItruncation));
if truncate_latency_slices < size(EEG.data,2)
    truncate_latency = truncate_latency_slices/EEG.srate;
    EEG = pop_select(EEG, 'notime',[double(truncate_latency) (size(EEG.data,2)/EEG.srate)]);
end

%% Remove ECG channel:
EEG = pop_select(EEG, 'nochannel',scan_parameters.ECG_channel);

%% Preprocess to remove other noise and reject ICA noise artifacts:

curr_dataset_name = ['task_' dataset_name '_' block_type '_TaskBlock' num2str(m)];
if overwrite_files || isempty(dir([task_dir filesep 'PreProcessed' filesep curr_dataset_name '_preprocessed.set']))
    EEG = offline_preprocess_manual_compiled(offline_preprocess_cfg,task_dir,curr_dataset_name,overwrite_files,EEG,base_path);
    % EEG = offline_preprocess_auto(offline_preprocess_cfg,task_dir,curr_dataset_name,overwrite_files,EEG);
else
    EEG = pop_loadset('filename',[curr_dataset_name '_preprocessed.set'],'filepath',[task_dir filesep 'PreProcessed']); EEG = eeg_checkset( EEG );
end