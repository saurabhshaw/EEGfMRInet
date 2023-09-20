
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
EEG = cust_loadbv(file_path, file_name);
EEG.setname = dataset_name;
EEG = eeg_checkset( EEG );

%% Check if triggers are correct:
Trigs_init = [];
for E = 1:length(EEG.event)
    if strcmp(EEG.event(E).type,slice_Marker)
        Trigs_init(end+1)=round(EEG.event(E).latency);
    end
end

abnormal_trigs = false;

if curr_func_size % If the scan is the larger one (300 time points)
    
    % Check if has extra markers:
    if (length(Trigs_init) > (num_Slices*num_Temporal_pts))
        abnormal_trigs = true;
        fastr_param.Trigs = Trigs_init(1:num_Slices*num_Temporal_pts);
    end
    
elseif ~curr_func_size % If the scan is the smaller one (150 time points)
    
    % Check if has extra markers:
    if (length(Trigs_init) > (num_Slices*num_Temporal_pts))
        abnormal_trigs = true;
        fastr_param.Trigs = Trigs_init(length(Trigs_init) - num_Slices*num_Temporal_pts+1:end);
    end
end

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
    
    EEG.setname = strcat(EEG.setname,'_GArem'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_FASTR'),'filepath',file_path);

else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_FASTR.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Identify and Label the QRS peaks:
curr_files_qrs = dir(strcat(dataset_name,'_backup_QRS*'));
if isempty(curr_files_qrs) % Check if backup present
    
    % Median Filter the ECG channel to remove any spurious peaks:
    % original_ECGchan = EEG.data(ECG_channel,:);
    % EEG.data(ECG_channel,:) = medfilt1(EEG.data(ECG_channel,:),100);
    
    EEG = pop_fmrib_qrsdetect(EEG,ECG_channel,'qrs','no');
    
    EEG.setname = strcat(EEG.setname,'_QRS'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_QRS'),'filepath',file_path);
    
else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_QRS.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Filter the Ballistocardiogram artifacts using the identified QRS peaks:
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
    
    EEG.setname = strcat(EEG.setname,'rem'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_PAS'),'filepath',file_path);
    
else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_PAS.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Resample the data to the specified sampling rate:
curr_files_resamp = dir(strcat(dataset_name,'_backup_RESAMP*'));
if isempty(curr_files_resamp) % Check if backup present
    EEG = pop_resample_cust(EEG, low_FS);
    
    EEG.setname = strcat(EEG.setname,'_resamp',num2str(low_FS)); EEG = eeg_checkset( EEG );    
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_RESAMP'),'filepath',file_path);

else % If Backup already present - Load the backup file    
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_RESAMP.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Filter the data using a bandpass filter:
if EEGLAB_preprocess_BPfilter
    curr_files_filt = dir(strcat(dataset_name,'_backup_FILT*'));
    if isempty(curr_files_filt) % Check if backup present
        EEG = pop_eegfiltnew(EEG, low_bp_filt, high_bp_filt, 1650, 0, [], 1);
        
        EEG.setname = strcat(EEG.setname,'_bpfilt'); EEG = eeg_checkset( EEG );        
        EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_FILT'),'filepath',file_path);
        
    else % If Backup already present - Load the backup file
        EEG = pop_loadset('filename',strcat(dataset_name,'_backup_FILT.set'),'filepath',file_path);
        EEG = eeg_checkset( EEG );
    end
end

%% Check for NaN values in the electrodes - if present, interpolate that channel:
if (sum(isnan(EEG.data(:))) > 0)
    curr_files_interp = dir(strcat(dataset_name,'_backup_NANINTERP*'));
    if isempty(curr_files_interp)
        
        % Add electrode locations:
        electrode_locations = load('Electrode_locations_final.mat', 'EEG');
        EEG.chanlocs = electrode_locations.EEG.chanlocs;
        EEG = eeg_checkset( EEG );
        
        EEG_isnan = isnan(EEG.data);
        EEG_isnan_ind = find(sum(EEG_isnan,2) > 0);
        EEG = pop_interp(EEG, EEG_isnan_ind, 'spherical');
        
        EEG.setname = strcat(EEG.setname,'_nanInterp'); EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_NANINTERP'),'filepath',file_path);
        
    else % If Backup already present - Load the backup file
        EEG = pop_loadset('filename',strcat(dataset_name,'_backup_NANINTERP.set'),'filepath',file_path);
        EEG = eeg_checkset( EEG );
    end
end

%% Detect bad channels:
% Using spectrum criteria and 3SDeviations as channel outlier threshold, done twice
curr_files_badchans = dir(strcat(dataset_name,'_backup_BADCHAN*'));
if isempty(curr_files_badchans) % Check if backup present
    
    % Add electrode locations:
    electrode_locations = load('Electrode_locations_final.mat', 'EEG');
    EEG.chanlocs = electrode_locations.EEG.chanlocs;
    EEG = eeg_checkset( EEG );
    
    full_channel_locations = EEG.chanlocs;
    EEG = pop_rejchan(EEG, 'elec',[1:size(EEG.data,1)],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
    EEG = eeg_checkset( EEG );
    
    EEG = pop_rejchan(EEG, 'elec',[1:EEG.nbchan],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
    EEG = eeg_checkset( EEG );
    selected_channel_locations = EEG.chanlocs; selected_channel_labels = {selected_channel_locations.labels};
    bad_channels_removed = setdiff({full_channel_locations(:).labels}, selected_channel_labels);
    
    EEG.setname = strcat(EEG.setname,'_badchan'); EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_BADCHAN'),'filepath',file_path);

else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_BADCHAN.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Select data around triggers to remove artifactual data for ICA:
if ICA_data_select
    Slice_marker_latencies = [EEG.event(find(strcmp(slice_Marker,{EEG.event.type}))).latency];
    startidx = min(Slice_marker_latencies) - ICA_data_select_range*EEG.srate; endidx = max(Slice_marker_latencies) + ICA_data_select_range*EEG.srate;
    if startidx < 1 startidx = 1; end
    if endidx > size(EEG.data,2) endidx = size(EEG.data,2); end
    
    EEG = pop_select(EEG,'point',[startidx endidx]);
end


%% Run ICA and reject components that are not Brain or Other (according to ICLabel):
curr_files_ICA = dir(strcat(dataset_name,'_backup_ICAreject*'));
if isempty(curr_files_ICA) % Check if backup present
    EEG = pop_runica(EEG, 'extended',1); EEG = eeg_checkset( EEG );
    EEG = pop_iclabel(EEG,'default'); EEG = eeg_checkset( EEG );
    EEG_ICLabel = EEG.etc.ic_classification.ICLabel.classifications;
    [~,EEG_ICLabel_max] = max(EEG_ICLabel,[],2);
    EEG_ICLabel_max_bin = (1 < EEG_ICLabel_max) & (EEG_ICLabel_max < 7);
    
    EEG_ICLabel_max_bin_final = EEG_ICLabel_max_bin;
    for i = 1:length(EEG_ICLabel_max_bin)
        if EEG_ICLabel_max_bin(i) == 1
            EEG_ICLabel_curr_brain = EEG_ICLabel(EEG_ICLabel_max_bin(i),1);
            EEG_ICLabel_curr_max = EEG_ICLabel(EEG_ICLabel_max_bin(i),EEG_ICLabel_max(i));
            if (EEG_ICLabel_curr_brain + 0.025) >= EEG_ICLabel_curr_max
                EEG_ICLabel_max_bin_final(i) = 0;
            end
        end
    end
    
    components = find(EEG_ICLabel_max_bin);
    EEG = pop_subcomp_cust(EEG, components, 0);
    
    EEG.setname = [EEG.setname '_ICAreject']; EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_ICAreject'),'filepath',file_path);

else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_ICAreject.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% ReReference the data:
curr_files_reref = dir(strcat(dataset_name,'_backup_REREF*'));
if isempty(curr_files_reref) % Check if backup present
    EEG = pop_reref( EEG, []);
    
    EEG.setname = [EEG.setname '_reRef']; EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_REREF'),'filepath',file_path);
    
else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_REREF.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Interpolate bad channels identified earlier:
curr_files_badinterp = dir(strcat(dataset_name,'_backup_BADINTERP*'));
if isempty(curr_files_badinterp) % Check if backup present
    EEG = pop_interp(EEG, full_channel_locations, 'spherical');
    
    EEG.setname = [EEG.setname '_badInterp']; EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_backup_BADINTERP'),'filepath',file_path);
    
else % If Backup already present - Load the backup file
    EEG = pop_loadset('filename',strcat(dataset_name,'_backup_BADINTERP.set'),'filepath',file_path);
    EEG = eeg_checkset( EEG );
end

%% Save the preprocessed dataset:
EEG = pop_saveset( EEG, 'filename',strcat(dataset_name,'_preproc'),'filepath',file_path);
EEG = eeg_checkset( EEG );

% Resetting abnormal_trigs flag:
abnormal_trigs = false;

% % To plot the topographic map:  
% figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
% figure; pop_spectopo(EEG, 1, [0         315399.8], 'EEG' , 'percent', 10, 'freq', [6 10 22], 'freqrange',[2 25],'electrodes','off');
