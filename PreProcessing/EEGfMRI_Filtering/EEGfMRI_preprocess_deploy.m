function EEGfMRI_preprocess_deploy(input_mat)
% Loads the Raw EEG files into EEGLAB and preprocesses them to remove the
% Gradient and Ballistocardiogram Artifacts present in simultaneous
% EEG_fMRI recordings
%
% INPUT:
% output_dir (string)    : The full path of the folder which contains the data files
% file_name (string)    : The name of the file (with the extension .vhdr)
% dataset_name (string) : The name to be assigned to the EEGLAB dataset
% 
% OUTPUT:
% Saves the preprocessed data in an EEGLAB dataset with the name
% "*dataset_name*_preproc"
%
% SEE ALSO:
%
% Author: Saurabh Shaw
%

%% Read in data:
% [EEG] = EEGfMRI_preprocess_deploy(EEG,scan_parameters,output_dir,fastr_param,overwrite_files)
load(input_mat);

%% Setup parallel options:
if isempty(gcp('nocreate'))
    numCores = feature('numcores')
    p = parpool(numCores);
end

%% Setup parameters:
% overwrite_files = 0;

abnormal_trigs = 0;
dataset_name = EEG.setname;
num_Slices = scan_parameters.slicespervolume; % Number of slices per volume
num_Temporal_pts = scan_parameters.tfunc_num_images/scan_parameters.slicespervolume; % Number of temporal points acquired
ECG_channel = scan_parameters.ECG_channel; % The channel number of the ECG lead
TR = scan_parameters.TR; % Repetition time of the scan (in Seconds)
slice_Marker = scan_parameters.slice_marker; % The value of the slice marker in the data
low_FS = scan_parameters.low_srate;
fastr_param.exclude_chan = [scan_parameters.ECG_channel]; % Channels not to perform OBS on.
ICA_data_select = fastr_param.ICA_data_select; % Whether to prune data or not
ICA_data_select_range = fastr_param.ICA_data_select_range; % Number of seconds before and after the last slice marker to keep for ICA analysis
append_length = 1; % The length of time to append the EEG data before FASTR - to avoid index out of bounds errors
append_threshold = 0.5; % The minimum length of time after the last slice marker which will trigger the appending of extra data.

%% Make folder if not already made:
if ~isdir([output_dir filesep 'EEGfMRI_PreProcessed'])
    mkdir([output_dir filesep 'EEGfMRI_PreProcessed']);
end
output_dir = [output_dir filesep 'EEGfMRI_PreProcessed'];

%% Run the FASTR Gradient Artifact Removal:
curr_files_fastr = dir(strcat(output_dir,filesep,dataset_name,'_backup_FASTR*'));
if overwrite_files || isempty(curr_files_fastr) % Check if backup present    
    fastr_param.Volumes = num_Temporal_pts; % Number of temporal points acquired
    fastr_param.Slices = num_Slices; % Number of slices per volume
    
    % Check if enough samples after the end of the last slice, add if not:
    slice_latencies = [EEG.event(find(strcmp(scan_parameters.slice_marker,{EEG.event.type}))).latency]; 
    append_reqd = ((size(EEG.data,2) - slice_latencies(end))./EEG.srate) < append_threshold;
    original_EEG_length = size(EEG.data,2);
    if append_length && append_reqd
        EEG.data = padarray(EEG.data',append_length*EEG.srate,'post')';
    end            
    
    switch fastr_param.parallel
        case 'none'
            if ~abnormal_trigs % If triggers correct -
                EEG = pop_fmrib_fastr(EEG, fastr_param.lpf, fastr_param.L, fastr_param.window,...
                    slice_Marker, fastr_param.strig, fastr_param.anc_chk,...
                    fastr_param.tc_chk, fastr_param.Volumes, fastr_param.Slices,...
                    fastr_param.rel_pos, fastr_param.exclude_chan, fastr_param.num_PC);
            else
                EEG = fmrib_fastr(EEG, fastr_param.lpf, fastr_param.L, fastr_param.window,...
                    fastr_param.Trigs, fastr_param.strig, fastr_param.anc_chk,...
                    fastr_param.tc_chk, fastr_param.Volumes, fastr_param.Slices,...
                    fastr_param.rel_pos, fastr_param.exclude_chan, fastr_param.num_PC);
            end
            
        case 'cpu'
            if ~abnormal_trigs % If triggers correct -
                EEG = pop_fmrib_fastr_cpu(EEG, fastr_param.lpf, fastr_param.L, fastr_param.window,...
                    slice_Marker, fastr_param.strig, fastr_param.anc_chk,...
                    fastr_param.tc_chk, fastr_param.Volumes, fastr_param.Slices,...
                    fastr_param.rel_pos, fastr_param.exclude_chan, fastr_param.num_PC);
            else
                EEG = fmrib_fastr_cpu(EEG, fastr_param.lpf, fastr_param.L, fastr_param.window,...
                    fastr_param.Trigs, fastr_param.strig, fastr_param.anc_chk,...
                    fastr_param.tc_chk, fastr_param.Volumes, fastr_param.Slices,...
                    fastr_param.rel_pos, fastr_param.exclude_chan, fastr_param.num_PC);
            end
            
        case 'gpu'
            if ~abnormal_trigs % If triggers correct -
                EEG = pop_fmrib_fastr_gpu(EEG, fastr_param.lpf, fastr_param.L, fastr_param.window,...
                    slice_Marker, fastr_param.strig, fastr_param.anc_chk,...
                    fastr_param.tc_chk, fastr_param.Volumes, fastr_param.Slices,...
                    fastr_param.rel_pos, fastr_param.exclude_chan, fastr_param.num_PC);
            else
                EEG = fmrib_fastr_gpu(EEG, fastr_param.lpf, fastr_param.L, fastr_param.window,...
                    fastr_param.Trigs, fastr_param.strig, fastr_param.anc_chk,...
                    fastr_param.tc_chk, fastr_param.Volumes, fastr_param.Slices,...
                    fastr_param.rel_pos, fastr_param.exclude_chan, fastr_param.num_PC);
            end
            
    end
    
    % Remove padded samples if added before FASTR:
    if append_length && append_reqd
        EEG.data = EEG.data(:,1:original_EEG_length);
    end 
    
    EEG.setname = strcat(EEG.setname,'_GArem'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_FASTR'),'filepath',output_dir);

else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_FASTR.set'),'filepath',output_dir);
    EEG = eeg_checkset( EEG );
end

%% Identify and Label the QRS peaks:
curr_files_qrs = dir(strcat(output_dir,filesep,dataset_name,'_backup_QRS*'));
if overwrite_files || isempty(curr_files_qrs) % Check if backup present    
    try
        EEG = pop_fmrib_qrsdetect(EEG,ECG_channel,'qrs','no');
    catch
        % Remove the slices after the end and before the beginning of the fMRI scans to avoid errors in
        % peak detection due to boundary spikes:
        endpad_amount = 300; firstpad_amount = 100;
        curr_num_slice_markers = find(cellfun(@(x)strcmp(x,scan_parameters.slice_marker),{EEG.event(:).type}));
        last_slice_latency = EEG.event(curr_num_slice_markers(end)).latency + endpad_amount;
        [EEG] = pop_select(EEG, 'notime',[last_slice_latency./EEG.srate size(EEG.data,2)./EEG.srate]);
        
        first_slice_latency = EEG.event(curr_num_slice_markers(1)).latency - firstpad_amount;
        if (first_slice_latency <= 0) first_slice_latency = EEG.event(curr_num_slice_markers(1)).latency; end
        [EEG] = pop_select(EEG, 'notime',[0 first_slice_latency./EEG.srate]);
        
        EEG = pop_fmrib_qrsdetect(EEG,ECG_channel,'qrs','no');
    end
    EEG.setname = strcat(EEG.setname,'_QRSid'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_QRS'),'filepath',output_dir);
    
else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_QRS.set'),'filepath',output_dir);
    EEG = eeg_checkset( EEG );
end

%% Filter the Ballistocardiogram artifacts using the identified QRS peaks:
curr_files_pas = dir(strcat(output_dir,filesep,dataset_name,'_backup_PAS*'));
if overwrite_files || isempty(curr_files_pas) % Check if backup present
    switch fastr_param.parallel
        case 'none'
            EEG = pop_fmrib_pas(EEG,'qrs','obs',4);
            
        case 'cpu'
            EEG = pop_fmrib_pas_cpu(EEG,'qrs','obs',4);
            
        case 'gpu'
            EEG = pop_fmrib_pas_gpu(EEG,'qrs','obs',4);
    end
    EEG.setname = strcat(EEG.setname,'_QRSrem'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_PAS'),'filepath',output_dir);
    
else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_PAS.set'),'filepath',output_dir);
    EEG = eeg_checkset( EEG );
end

%% Resample the data to the specified sampling rate:
curr_files_resamp = dir(strcat(output_dir,filesep,dataset_name,'_backup_RESAMP*'));
if overwrite_files || isempty(curr_files_resamp) % Check if backup present
    EEG = pop_resample(EEG, low_FS);
    EEG.setname = regexprep(EEG.setname,' resampled',strcat('_resamp',num2str(low_FS))); EEG = eeg_checkset( EEG );
    % EEG.setname = strcat(EEG.setname,'_resamp',num2str(low_FS)); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_RESAMP'),'filepath',output_dir);

else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_RESAMP.set'),'filepath',output_dir);
    EEG = eeg_checkset( EEG );
end

% Filter the data using a bandpass filter:
if fastr_param.EEGLAB_preprocess_BPfilter
    curr_files_filt = dir(strcat(output_dir,filesep,dataset_name,'_backup_FILT*'));
    if overwrite_files || isempty(curr_files_filt) % Check if backup present
        EEG = pop_eegfiltnew(EEG, fastr_param.low_bp_filt, fastr_param.high_bp_filt, 1650, 0, [], 1);
        EEG.setname = strcat(EEG.setname,'_bpfilt');
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_FILT'),'filepath',output_dir);
        
    else % If Backup already present - Load the backup file
        EEG = pop_loadset('filename',strcat(dataset_name,'_backup_FILT.set'),'filepath',output_dir);
        EEG = eeg_checkset( EEG );
    end
end

%% Check for NaN values in the electrodes - if present, interpolate that channel:
% NaNs are caused by the ANC in FASTR filtering - add in code that reruns
% FASTR for those electrodes that have NaNs in them rather than
% interpolation, by changing EEGfMRI_preprocess_param.anc_chk = 0;

if (sum(isnan(EEG.data(:))) > 0)
    curr_files_interp = dir(strcat(output_dir,filesep,dataset_name,'_backup_INTERP*'));
%    if overwrite_files || isempty(curr_files_interp)
        
%         % Add electrode locations:
%         electrode_locations = load('Electrode_locations_final.mat', 'EEG');
%         EEG.chanlocs = electrode_locations.EEG.chanlocs;
%         EEG = eeg_checkset( EEG );
        
        EEG_isnan = isnan(EEG.data);
        EEG_isnan_ind = find(sum(EEG_isnan,2) > 0);
        if length(EEG_isnan_ind) < ceil(0.15*size(EEG.data,1))
            EEG_isnan_ind_str = '';
            for i = 1:length(EEG_isnan_ind) EEG_isnan_ind_str = [EEG_isnan_ind_str ' ' num2str(EEG_isnan_ind(i)) ' ']; end
            fprintf(['\n Interpolating NaN electrodes' EEG_isnan_ind_str ' \n']);
            EEG = pop_interp(EEG, EEG_isnan_ind, 'spherical');
            EEG.setname = strcat(EEG.setname,'_interp'); EEG = eeg_checkset( EEG );
            EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_INTERP'),'filepath',output_dir);
        else
            fprintf(['\n FAULT - Too many electrodes with NaNs - Will be removed with segments \n']);
        end

        
%    else % If Backup already present - Load the backup file
%        EEG = pop_loadset('filename',strcat(dataset_name,'_backup_INTERP.set'),'filepath',output_dir);
%        EEG = eeg_checkset( EEG );
%    end
end

%% Select data around triggers to remove artifactual data for ICA:
if ICA_data_select
    Slice_marker_latencies = [EEG.event(find(strcmp(slice_Marker,{EEG.event.type}))).latency];
    startidx = max(min(Slice_marker_latencies) - ICA_data_select_range*EEG.srate,1);
    endidx = min(max(Slice_marker_latencies) + ICA_data_select_range*EEG.srate,length(EEG.times));
    if startidx < 1 startidx = 1; end
    if endidx > size(EEG.data,2) endidx = size(EEG.data,2); end
    
    EEG = pop_select(EEG,'point',[startidx endidx]);
end

%% Save the preprocessed dataset:
EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_EEGfMRIpreprocessed'),'filepath',output_dir);
EEG = eeg_checkset( EEG );

% Resetting abnormal_trigs flag:
% abnormal_trigs = false;

% % To plot the topographic map:  
% figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
% figure; pop_spectopo(EEG, 1, [0         315399.8], 'EEG' , 'percent', 10, 'freq', [6 10 22], 'freqrange',[2 25],'electrodes','off');
