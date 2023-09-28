function [EEG] = EEGfMRI_preprocess_full(EEG,condition_dir,scan_param,participant_id,curr_condition,num_volumes,offline_preprocess_cfg,EEGfMRI_preprocess_param,overwrite_files,base_path)
% Loads the Raw EEG files into EEGLAB and preprocesses them to remove the
% Gradient and Ballistocardiogram Artifacts present in simultaneous
% EEG_fMRI recordings
%
% INPUT:
% output_dir (string)    : The full path of the folder which contains the data files
% file_name (string)    : The name of the file (with the extension .vhdr)
% EEG.setname (string) : The name to be assigned to the EEGLAB dataset
% 
% OUTPUT:
% Saves the preprocessed data in an EEGLAB dataset with the name
% "*EEG.setname*_preproc"
%
% SEE ALSO:
%
% Author: Saurabh Shaw
%

if overwrite_files || isempty(dir([condition_dir filesep 'EEGfMRI_PreProcessed' filesep EEG.setname '_EEGfMRIpreprocessed.set']))
    %% Make folder if not already made:
    if ~isfolder([condition_dir filesep 'EEGfMRI_PreProcessed'])
        mkdir([condition_dir filesep 'EEGfMRI_PreProcessed']);
    end
    condition_dir_eegfmri_preprocess = [condition_dir filesep 'EEGfMRI_PreProcessed'];
    
    %% Run the FASTR Gradient Artifact Removal:
    curr_files_fastr = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_backup_FASTR*'));
    if overwrite_files || isempty(curr_files_fastr) % Check if backup present    
        % Check if enough samples after the end of the last slice, add if not:
        slice_latencies = [EEG.event((find([EEG.event.type]==scan_param.slice_marker))).latency];
        append_reqd = ((size(EEG.data,2) - slice_latencies(end))./EEG.srate) < EEGfMRI_preprocess_param.fastr_data_append_threshold;
        % Store original length to remove padding after FASTR
        original_EEG_length = size(EEG.data,2);
        if append_reqd
            padding = zeros(size(EEG.data,1),EEGfMRI_preprocess_param.fastr_data_append_length*EEG.srate);
            EEG.data = [EEG.data padding];
        end            

        for E=1:length(EEG.event)
            if strcmp(num2str(EEG.event(E).type),num2str(scan_param.slice_marker))
                EEGfMRI_preprocess_param.Trigs(end+1)=round(EEG.event(E).latency);
            end
        end

        if EEGfMRI_preprocess_param.use_fastr_gui
            EEG = pop_fmrib_fastr(EEG,EEGfMRI_preprocess_param);
        else     
            switch EEGfMRI_preprocess_param.parallel
                case 'none'
                    EEG = fmrib_fastr(EEG, EEGfMRI_preprocess_param.lpf, EEGfMRI_preprocess_param.L, EEGfMRI_preprocess_param.window,...
                        EEGfMRI_preprocess_param.Trigs, EEGfMRI_preprocess_param.strig, EEGfMRI_preprocess_param.anc_chk,...
                        EEGfMRI_preprocess_param.tc_chk, num_volumes, scan_param.slicespervolume,...
                        EEGfMRI_preprocess_param.rel_pos, EEGfMRI_preprocess_param.exclude_chan, EEGfMRI_preprocess_param.num_PC);
                case 'cpu'
                    EEG = fmrib_fastr_cpu(EEG, EEGfMRI_preprocess_param.lpf, EEGfMRI_preprocess_param.L, EEGfMRI_preprocess_param.window,...
                        EEGfMRI_preprocess_param.Trigs, EEGfMRI_preprocess_param.strig, EEGfMRI_preprocess_param.anc_chk,...
                        EEGfMRI_preprocess_param.tc_chk, num_volumes, scan_param.slicespervolume,...
                        EEGfMRI_preprocess_param.rel_pos, EEGfMRI_preprocess_param.exclude_chan, EEGfMRI_preprocess_param.num_PC);
                case 'gpu' % in development  
                    EEG = fmrib_fastr_gpu(EEG, EEGfMRI_preprocess_param.lpf, EEGfMRI_preprocess_param.L, EEGfMRI_preprocess_param.window,...
                        EEGfMRI_preprocess_param.Trigs, EEGfMRI_preprocess_param.strig, EEGfMRI_preprocess_param.anc_chk,...
                        EEGfMRI_preprocess_param.tc_chk, num_volumes, scan_param.slicespervolume,...
                        EEGfMRI_preprocess_param.rel_pos, EEGfMRI_preprocess_param.exclude_chan, EEGfMRI_preprocess_param.num_PC);    
            end
        end
        
        % Remove padded samples if added before FASTR:
        if append_reqd
            EEG.data = EEG.data(:,1:original_EEG_length);
        end 
        
        EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_backup_FASTR'),'filepath',condition_dir_eegfmri_preprocess);
    
    else % If Backup already present - Load the backup file    
        EEG = pop_loadset('filename',strcat(EEG.setname,'_backup_FASTR.set'),'filepath',condition_dir_eegfmri_preprocess);
        EEG = eeg_checkset( EEG );
    end
    
    %% Identify and Label the QRS peaks:
    curr_files_qrs = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_backup_QRS*'));
    if overwrite_files || isempty(curr_files_qrs) % Check if backup present    
        EEG = pop_fmrib_qrsdetect(EEG,scan_param.ECG_channel,EEGfMRI_preprocess_param.qrs_event_marker,'yes');
        EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_backup_QRS'),'filepath',condition_dir_eegfmri_preprocess);
        
    else % If Backup already present - Load the backup file
        EEG = pop_loadset('filename',strcat(EEG.setname,'_backup_QRS.set'),'filepath',condition_dir_eegfmri_preprocess);
        EEG = eeg_checkset( EEG );
    end
    
    %% Filter the Ballistocardiogram artifacts using the identified QRS peaks:
    curr_files_pas = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_backup_PAS*'));
    if overwrite_files || isempty(curr_files_pas) % Check if backup present
        if EEGfMRI_preprocess_param.use_pas_gui
            EEG = pop_fmrib_pas(EEG,EEGfMRI_preprocess_param);
        else

            QRSevents=[];
            for E=1:length(EEG.event)
                if strcmp(EEG.event(E).type,EEGfMRI_preprocess_param.qrs_event_marker)
                    QRSevents(end+1)=round(EEG.event(E).latency);
                end
            end

            switch EEGfMRI_preprocess_param.parallel
                case 'none'
                    EEG = fmrib_pas(EEG,QRSevents,'obs',4);

                case 'cpu'
                    EEG = fmrib_pas_cpu(EEG,QRSevents,'obs',4);

                case 'gpu'
                    EEG = fmrib_pas_gpu(EEG,QRSevents,'obs',4);
            end
        end
        EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_backup_PAS'),'filepath',condition_dir_eegfmri_preprocess);
        
    else % If Backup already present - Load the backup file    
        EEG = pop_loadset('filename',strcat(EEG.setname,'_backup_PAS.set'),'filepath',condition_dir_eegfmri_preprocess);
        EEG = eeg_checkset( EEG );
    end
    
    %% Resample the data to the specified sampling rate:
    curr_files_resamp = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_backup_RESAMP*'));
    if overwrite_files || isempty(curr_files_resamp) % Check if backup present
        EEG = pop_resample(EEG, scan_param.low_srate);
        EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_backup_RESAMP'),'filepath',condition_dir_eegfmri_preprocess);
    
    else % If Backup already present - Load the backup file    
        EEG = pop_loadset('filename',strcat(EEG.setname,'_backup_RESAMP.set'),'filepath',condition_dir_eegfmri_preprocess);
        EEG = eeg_checkset( EEG );
    end

    %% Check for NaN values in the electrodes - interpolate if req'd
    if (sum(isnan(EEG.data(:))) > 0)
        curr_files_interp = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_INTERP*'));
        if isempty(curr_files_interp) || control_param.overwrite_files
            EEG_isnan = isnan(EEG.data);
            EEG_isnan_ind = find(sum(EEG_isnan,2) > 0);
            if length(EEG_isnan_ind) < ceil(0.10*size(EEG.data,1))
                EEG_isnan_ind_str = '';
                for i = 1:length(EEG_isnan_ind)
                    EEG_isnan_ind_str = [EEG_isnan_ind_str ' ' num2str(EEG_isnan_ind(i)) ' '];
                end
                fprintf(['\n Interpolating NaN electrodes' EEG_isnan_ind_str ' \n']);
                EEG = pop_interp(EEG, EEG_isnan_ind, 'spherical');
                EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_INTERP'),'filepath',condition_dir_eegfmri_preprocess);
            else
                fprintf('\n FAULT - Too many electrodes with NaNs - Will be removed with segments \n');
            end
        else % If Backup already present - Load the backup file
            EEG = pop_loadset('filename',strcat(EEG.setname,'_INTERP.set'),'filepath',condition_dir_eegfmri_preprocess);
            EEG = eeg_checkset( EEG );
        end
    end
    
    % %% Filter the data using a bandpass filter:
    % if EEGfMRI_preprocess_param.EEGLAB_preprocess_BPfilter
    %     curr_files_filt = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_backup_bpFILT*'));
    %     if overwrite_files || isempty(curr_files_filt) % Check if backup present
    %         %bp filtering (isolating eeg)
    %         EEG = pop_eegfiltnew(EEG, EEGfMRI_preprocess_param.low_bp_filt, EEGfMRI_preprocess_param.high_bp_filt, 16500, 0, [], 1);
    %         EEG = eeg_checkset( EEG );
    %         EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_backup_bpFILT'),'filepath',condition_dir_eegfmri_preprocess);
    %     else % If Backup already present - Load the backup file
    %         EEG = pop_loadset('filename',strcat(EEG.setname,'_backup_bpFILT.set'),'filepath',condition_dir_eegfmri_preprocess);
    %         EEG = eeg_checkset( EEG );
    %     end
    % end
    % 
    % %% Filter the data using notch filter(s):dev in progress
    % if EEGfMRI_preprocess_param.EEGLAB_preprocess_Nfilter
    %     curr_files_filt = dir(strcat(condition_dir_eegfmri_preprocess,filesep,EEG.setname,'_backup_nFILT_bpFILT*'));
    %     if overwrite_files || isempty(curr_files_filt) % Check if backup present     
    %         %notch filtering (removing ac power artifacts, slice artifacts)
    %         % EEG = 
    %         EEG = eeg_checkset( EEG );
    %         EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_backup_nFILT_bpFILT'),'filepath',condition_dir_eegfmri_preprocess);
    %     else % If Backup already present - Load the backup file
    %         EEG = pop_loadset('filename',strcat(EEG.setname,'_backup_nFILT_bpFILT.set'),'filepath',condition_dir_eegfmri_preprocess);
    %         EEG = eeg_checkset( EEG );
    %     end
    % end

    %% Select data around triggers to remove artifactual data for ICA:
    if EEGfMRI_preprocess_param.ICA_data_select
        % Note that EEG event type datatype is string after qrs detection
        % therefore cannot use the slice_latencies logic seen line 34
        slice_latencies = [EEG.event((find(strcmp({EEG.event.type},num2str(scan_param.slice_marker))))).latency];
        startidx = max(min(slice_latencies) - EEGfMRI_preprocess_param.ICA_data_select_range*EEG.srate,1);
        endidx = min(max(slice_latencies) + EEGfMRI_preprocess_param.ICA_data_select_range*EEG.srate,length(EEG.times));
        EEG = pop_select(EEG,'point',[startidx endidx]);
    end
    
    %% Save the preprocessed dataset:
    EEG = pop_saveset( EEG, 'filename',strcat(EEG.setname,'_EEGfMRIpreprocessed'),'filepath',condition_dir_eegfmri_preprocess);
    EEG = eeg_checkset( EEG );
    
    %finished eegfmri-specific preprocessing - onto general preprocessing
    
else
    EEG = pop_loadset('filename',[EEG.setname '_EEGfMRIpreprocessed.set'],'filepath',[condition_dir filesep 'EEGfMRI_PreProcessed']); EEG = eeg_checkset( EEG );
end

%% Preprocess to remove other noise and reject ICA noise artifacts:

if overwrite_files || isempty(dir([condition_dir filesep 'PreProcessed' filesep EEG.setname '_preprocessed.set']))
    EEG = offline_preprocess_manual_deploy(offline_preprocess_cfg,condition_dir,EEG.setname,overwrite_files,EEG);
else
    EEG = pop_loadset('filename',[EEG.setname '_preprocessed.set'],'filepath',[condition_dir filesep 'PreProcessed']); EEG = eeg_checkset( EEG );
end

%% Prep for feature extraction: Remove data before first trigger and after last trigger +TR secs
slice_latencies = [EEG.event((find(strcmp({EEG.event.type},num2str(scan_param.slice_marker))))).latency];
startidx = max(min(slice_latencies)-1,1);
endidx = min(max(slice_latencies) + scan_param.TR*EEG.srate-1,length(EEG.times));
EEG = pop_select(EEG,'point',[startidx endidx]);
EEG = pop_saveset(EEG,'filename','ready_for_feature_extraction','filepath',[condition_dir filesep 'PreProcessed']);
