function [EEG] = EEGfMRI_preprocess(EEG,scan_parameters)
% Loads the Raw EEG files into EEGLAB and preprocesses them to remove the
% Gradient and Ballistocardiogram Artifacts present in simultaneous
% EEG_fMRI recordings
%
% INPUT:
% file_path (string)    : The full path of the folder which contains the data files
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


% Import Brain Vision Files:
% EEG = cust_loadbv(file_path, file_name);
dataset_name = EEG.setname;

% % Check if triggers are correct:
% Trigs_init = [];
% for E = 1:length(EEG.event)
%     if strcmp(EEG.event(E).type,slice_Marker)
%         Trigs_init(end+1)=round(EEG.event(E).latency);
%     end
% end
% 
% abnormal_trigs = false;
% 
% if curr_func_size % If the scan is the larger one (300 time points)    
%     % Check if has extra markers:
%     if (length(Trigs_init) > (num_Slices*num_Temporal_pts))
%         abnormal_trigs = true;
%         fastr_param.Trigs = Trigs_init(1:num_Slices*num_Temporal_pts);
%     end
%     
% elseif ~curr_func_size % If the scan is the smaller one (150 time points)
%     
%     % Check if has extra markers:
%     if (length(Trigs_init) > (num_Slices*num_Temporal_pts))
%         abnormal_trigs = true;
%         fastr_param.Trigs = Trigs_init(length(Trigs_init) - num_Slices*num_Temporal_pts+1:end);
%     end
% end

%% Run the FASTR Gradient Artifact Removal:
curr_files_fastr = dir(strcat(dataset_name,'_backup_FASTR*'));
if isempty(curr_files_fastr) % Check if backup present    
    fastr_param.Volumes = num_Temporal_pts; % Number of temporal points acquired
    fastr_param.Slices = num_Slices; % Number of slices per volume
    switch parallel
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
    EEG.setname = strcat(dataset_name,'_GA_rem');
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_FASTR'),'filepath',file_path);

else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_FASTR.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

% Identify and Label the QRS peaks:
curr_files_qrs = dir(strcat(dataset_name,'_backup_QRS*'));
if isempty(curr_files_qrs) % Check if backup present
    EEG = pop_fmrib_qrsdetect(EEG,ECG_channel,'qrs','no');
    EEG.setname = strcat(dataset_name,'_GA_rem_QRS');
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_QRS'),'filepath',file_path);
    
else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_QRS.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

% Filter the Ballistocardiogram artifacts using the identified QRS peaks:
curr_files_pas = dir(strcat(dataset_name,'_backup_PAS*'));
if isempty(curr_files_pas) % Check if backup present
    switch parallel
        case 'none'
            EEG = pop_fmrib_pas(EEG,'qrs','obs',4);
            
        case 'cpu'
            EEG = pop_fmrib_pas_cpu(EEG,'qrs','obs',4);
            
        case 'gpu'
            EEG = pop_fmrib_pas_gpu(EEG,'qrs','obs',4);
    end
    EEG.setname = strcat(dataset_name,'_GA_rem_QRS_rem');
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_PAS'),'filepath',file_path);
    
else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_PAS.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

% Resample the data to the specified sampling rate:
curr_files_resamp = dir(strcat(dataset_name,'_backup_RESAMP*'));
if isempty(curr_files_resamp) % Check if backup present
    EEG = pop_resample(EEG, low_FS);
    EEG.setname = strcat(dataset_name,'_GA_rem_QRS_rem_resamp_',num2str(low_FS));
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_RESAMP'),'filepath',file_path);

else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_RESAMP.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

% Filter the data using a bandpass filter:
if EEGLAB_preprocess_BPfilter
    curr_files_filt = dir(strcat(dataset_name,'_backup_FILT*'));
    if isempty(curr_files_filt) % Check if backup present
        EEG = pop_eegfiltnew(EEG, low_bp_filt, high_bp_filt, 1650, 0, [], 1);
        EEG.setname = strcat(dataset_name,'_GA_rem_QRS_rem_resamp_',num2str(low_FS),'filt');
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_FILT'),'filepath',file_path);
        
    else % If Backup already present - Load the backup file
        EEG = pop_loadset('filename',strcat(dataset_name,'_backup_FILT.set'),'filepath',file_path);
        EEG = eeg_checkset( EEG );
    end
end

% Check for NaN values in the electrodes - if present, interpolate that channel:
if (sum(isnan(EEG.data(:))) > 0)
    curr_files_interp = dir(strcat(dataset_name,'_backup_INTERP*'));
    if isempty(curr_files_interp)
        
        % Add electrode locations:
        electrode_locations = load('Electrode_locations_final.mat', 'EEG');
        EEG.chanlocs = electrode_locations.EEG.chanlocs;
        EEG = eeg_checkset( EEG );
        
        EEG_isnan = isnan(EEG.data);
        EEG_isnan_ind = find(sum(EEG_isnan,2) > 0);
        EEG = pop_interp(EEG, EEG_isnan_ind, 'spherical');
        EEG.setname = strcat(dataset_name,'_GA_rem_QRS_rem_resamp_',num2str(low_FS),'filt','_interp');
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_INTERP'),'filepath',file_path);
        
    else % If Backup already present - Load the backup file
        EEG = pop_loadset('filename',strcat(dataset_name,'_backup_INTERP.set'),'filepath',file_path);
        EEG = eeg_checkset( EEG );
    end
end

% Save the preprocessed dataset:
EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_preproc'),'filepath',file_path);
EEG = eeg_checkset( EEG );

% Resetting abnormal_trigs flag:
% abnormal_trigs = false;

% % To plot the topographic map:  
% figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
% figure; pop_spectopo(EEG, 1, [0         315399.8], 'EEG' , 'percent', 10, 'freq', [6 10 22], 'freqrange',[2 25],'electrodes','off');
