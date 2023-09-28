function EEG = offline_preprocess_manual_deploy(cfg,curr_dir,dataset_name,overwrite_files,EEG)

%% Make folder if not already made:
if ~isdir([curr_dir filesep 'PreProcessed'])
    mkdir([curr_dir filesep 'PreProcessed']);
end
curr_dir_preprocessed = [curr_dir filesep 'PreProcessed'];

%% Check status of the steps performed:
num_steps = 10; % Number of steps in the preprocessing pipeline
stageCompletion_file = [curr_dir_preprocessed filesep dataset_name '_StageCompletion.mat'];
if isempty(dir(stageCompletion_file)) || overwrite_files
    preprocessing_stageCompletion = zeros(1,num_steps); max_finishedStage = 0;
    save(stageCompletion_file);
else
    load(stageCompletion_file)
    % max_finishedStage = max(find(preprocessing_stageCompletion));
end
    
%% 1 - Filter the data:
current_stage = 1;
if max_finishedStage == current_stage-1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    EEG = pop_eegfiltnew(EEG,cfg.filter_lp,cfg.filter_hp); EEG.setname = [EEG.setname '_filt']; EEG = eeg_checkset( EEG );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% 2 - Detect bad channels:
current_stage = 2;
if max_finishedStage == current_stage-1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if cfg.remove_electrodes
        full_channel_locations = EEG.chanlocs;
        
        % Detect and remove flatline channels +
        % Filter out drifts in channels +
        % Remove channels that are not similar to other channels (based on 'ChannelCriterion')
        
        % [EEG,HP,BUR] = clean_artifacts(EEG,'ChannelCriterion',ChannelCriterion,'BurstCriterion','off','WindowCriterion','off');
        
        EEG = clean_flatlines_mod(EEG);

        % EEG = clean_flatlines(EEG,max_flatline_duration,max_allowed_jitter);
        % EEG = clean_drifts(EEG,drift_highpass_band);
        
        % Using spectrum criteria and 3SDeviations as channel outlier threshold, done twice
        % EEG = pop_rejchan(EEG, 'elec',[1:size(EEG.data,1)],'threshold',[-3 3],'norm','on','measure','prob');
        % EEG = pop_rejchan(EEG, 'elec',[1:size(EEG.data,1)],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
        % EEG = eeg_checkset( EEG );
        
        %     EEG = pop_rejchan(EEG, 'elec',[1:EEG.nbchan],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
        %     EEG = eeg_checkset( EEG );
        
        % Save removed channels
        selected_channel_locations = EEG.chanlocs; selected_channel_labels = {selected_channel_locations.labels};
        output_struct.bad_channels_removed = setdiff({full_channel_locations(:).labels}, selected_channel_labels);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% 3 - Segment data according to data type
current_stage = 3;
if max_finishedStage == current_stage-1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cfg.segment_data
        EEG = pop_epoch(EEG, cfg.segment_markers, [cfg.task_segment_start cfg.task_segment_end], 'epochinfo', 'yes');
        EEG.setname = [EEG.setname '_segmented'];
        EEG = pop_saveset( EEG, 'filename',['Stage' num2str(current_stage) '-' EEG.setname '.set'],'filepath',curr_dir_preprocessed);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% 4 - Run ICA and reject components that are not Brain or Other (according to ICLabel):
current_stage = 4;
if max_finishedStage == current_stage-1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num_ICA_reruns = 2;
    output_struct.EEG_ICA = cell(1,num_ICA_reruns); output_struct.EEG_ICA_IClabelrejcomp = cell(1,num_ICA_reruns); EEG_ICA_Manualrejcomp = cell(1,num_ICA_reruns);
    
    EEG = pop_runica(EEG, 'extended',1,'interupt','on'); EEG = eeg_checkset( EEG );
    EEG = pop_iclabel(EEG,'default'); EEG = eeg_checkset( EEG );
    EEG_ICLabel = EEG.etc.ic_classification.ICLabel.classifications;
    [~,EEG_ICLabel_max] = max(EEG_ICLabel,[],2);
    EEG_ICLabel_max_bin = (1 < EEG_ICLabel_max) & (EEG_ICLabel_max < 7);
    
    EEG_ICLabel_max_bin_final = EEG_ICLabel_max_bin;
    for i = 1:length(EEG_ICLabel_max_bin)
        if EEG_ICLabel_max_bin(i) == 1
            EEG_ICLabel_curr_brain = EEG_ICLabel(i,1);
            EEG_ICLabel_curr_max = EEG_ICLabel(i,EEG_ICLabel_max(i));
            if (EEG_ICLabel_curr_brain + 0.025) >= EEG_ICLabel_curr_max
                EEG_ICLabel_max_bin_final(i) = 0;
            end
        end
    end
    
    output_struct.EEG_ICA{1} = EEG;
    
    components = find(EEG_ICLabel_max_bin_final);
    output_struct.EEG_ICA_IClabelrejcomp{1} = components;
    EEG = pop_subcomp(EEG, components, 0);
    EEG = pop_iclabel(EEG,'default'); EEG = eeg_checkset( EEG );
    
    EEG.setname = regexprep(EEG.setname,' pruned with ICA','_ICAreject1');
    EEG = pop_saveset( EEG, 'filename',['Stage' num2str(current_stage) '-' EEG.setname '.set'],'filepath',curr_dir_preprocessed);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
    
    % Save breakpoint after updating preprocessing_stageCompletion
    if cfg.manualICA_check
        save([curr_dir_preprocessed filesep dataset_name '_Stage' num2str(current_stage) '-Workspace'],'-regexp', '^(?!(current_stage|preprocessing_stageCompletion|max_finishedStage)$).'); % Save breakpoint without saving the "current_stage" variable
    end
end

%% 5 - Manually reject the remaining ICA components:
% User load Stage4 data into EEGLAB and manually reject any other ICA
% components using offline_preprocess_manual_ICAselectionStage

if ~cfg.manualICA_check
    current_stage = 5;
    
    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% 6 - ReRun ICA and reject components that are not Brain or Other (according to ICLabel):
current_stage = 6;
if max_finishedStage == current_stage-1
    if cfg.manualICA_check
        prevStage_iscomplete = dir([curr_dir_preprocessed filesep dataset_name '_Stage' num2str(current_stage-1) '-Workspace*']);
    else prevStage_iscomplete = 1; end
            
    if ~isempty(prevStage_iscomplete) % Check if previous stage is complete and then proceed with rest of the analysis
        if cfg.manualICA_check 
            load([prevStage_iscomplete.folder filesep prevStage_iscomplete.name]); 
            output_struct.EEG_ICA_Manualrejcomp = EEG_ICA_Manualrejcomp;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if cfg.run_second_ICA EEG = pop_runica(EEG, 'extended',1,'interupt','on'); EEG = eeg_checkset( EEG ); end
        EEG = pop_iclabel(EEG,'default'); EEG = eeg_checkset( EEG );
        EEG_ICLabel = EEG.etc.ic_classification.ICLabel.classifications;
        [~,EEG_ICLabel_max] = max(EEG_ICLabel,[],2);
        EEG_ICLabel_max_bin = (1 < EEG_ICLabel_max) & (EEG_ICLabel_max < 7);
        
        EEG_ICLabel_max_bin_final = EEG_ICLabel_max_bin;
        for i = 1:length(EEG_ICLabel_max_bin)
            if EEG_ICLabel_max_bin(i) == 1
                EEG_ICLabel_curr_brain = EEG_ICLabel(i,1);
                EEG_ICLabel_curr_max = EEG_ICLabel(i,EEG_ICLabel_max(i));
                if (EEG_ICLabel_curr_brain + 0.025) >= EEG_ICLabel_curr_max
                    EEG_ICLabel_max_bin_final(i) = 0;
                end
            end
        end
        
        output_struct.EEG_ICA{2} = EEG; 
        
        components = find(EEG_ICLabel_max_bin_final);
        output_struct.EEG_ICA_IClabelrejcomp{2} = components;
        EEG = pop_subcomp(EEG, components, 0);
        EEG = pop_iclabel(EEG,'default'); EEG = eeg_checkset( EEG );
        
        EEG.setname = regexprep(EEG.setname,' pruned with ICA','_ICAreject2');
        EEG = pop_saveset(EEG, 'filename',['Stage' num2str(current_stage) '-' EEG.setname '.set'],'filepath',curr_dir_preprocessed);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Update preprocessing_stageCompletion
        preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
        save(stageCompletion_file);
        
        % Save breakpoint after updating preprocessing_stageCompletion
        if cfg.manualICA_check
            save([curr_dir_preprocessed filesep dataset_name '_Stage' num2str(current_stage) '-Workspace'],'-regexp', '^(?!(current_stage|preprocessing_stageCompletion|max_finishedStage)$).');
        end
    else
        preprocessing_stageCompletion(current_stage-1) = 0; max_finishedStage = max(find(preprocessing_stageCompletion));
        save(stageCompletion_file);
    end
end

%% 7 - Manually reject the remaining ICA components:
% User load Stage6 data into EEGLAB and manually reject any other ICA
% components using offline_preprocess_manual_ICAselectionStage

if ~cfg.manualICA_check
    current_stage = 7;
    
    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% 8 - ReReference the data:
current_stage = 8;
if max_finishedStage == current_stage-1
    if cfg.manualICA_check
        prevStage_iscomplete = dir([curr_dir_preprocessed filesep dataset_name '_Stage' num2str(current_stage-1) '-Workspace*']);
    else prevStage_iscomplete = 1; end
        
    if ~isempty(prevStage_iscomplete) % Check if previous stage is complete and then proceed with rest of the analysis
        if cfg.manualICA_check 
            load([prevStage_iscomplete.folder filesep prevStage_iscomplete.name]);
            output_struct.EEG_ICA_Manualrejcomp = EEG_ICA_Manualrejcomp;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        EEG = pop_reref( EEG, []); EEG.setname = [EEG.setname '_reRef']; EEG = eeg_checkset( EEG );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        % Update preprocessing_stageCompletion
        preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
        save(stageCompletion_file);
        
    else
        preprocessing_stageCompletion(current_stage-1) = 0; max_finishedStage = max(find(preprocessing_stageCompletion));
        save(stageCompletion_file);
    end
end

%% 9 - Reject segments:
current_stage = 9;
if max_finishedStage == current_stage-1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cfg.segment_data
        reject_min_amp = -40; reject_max_amp = 40; pipeline_visualizations_semiautomated = 0;
        % EEG = pop_eegthresh(EEG,1,[1:EEG.nbchan] ,[reject_min_amp],[reject_max_amp],[EEG.xmin],[EEG.xmax],2,0);
        EEG = pop_jointprob(EEG,1,[1:EEG.nbchan],3,3,pipeline_visualizations_semiautomated,...
            0,pipeline_visualizations_semiautomated);
        output_struct.preprocess_segRej = EEG.reject.rejjp; % list of epochs rejected
        EEG = eeg_rejsuperpose(EEG, 1, 0, 1, 1, 1, 1, 1, 1);
        EEG = pop_rejepoch(EEG, [EEG.reject.rejglobal] ,0);
        EEG.setname = [EEG.setname '_segRej']; EEG = eeg_checkset( EEG );
        EEG = pop_saveset(EEG, 'filename',['Stage' num2str(current_stage) '-' EEG.setname '.set'],'filepath',curr_dir_preprocessed);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% 10 - Interpolate bad channels identified earlier:
current_stage = 10;
if max_finishedStage == current_stage-1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stage Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cfg.remove_electrodes && ~isempty(output_struct.bad_channels_removed)
        EEG = pop_interp(EEG, full_channel_locations, 'spherical');
        EEG.setname = [EEG.setname '_chanInterp']; EEG = eeg_checkset( EEG );
        EEG = pop_saveset(EEG, 'filename',['Stage' num2str(current_stage) '-' EEG.setname '.set'],'filepath',curr_dir_preprocessed);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    save(stageCompletion_file);
end

%% Save final pre-processed dataset:
if max_finishedStage == current_stage
    EEG = pop_saveset(EEG, 'filename',[dataset_name '_preprocessed.set'],'filepath',curr_dir_preprocessed);
    save([curr_dir_preprocessed filesep dataset_name '_preprocessed'],'-struct','output_struct');
%     if remove_electrodes && segment_data
%         save([curr_dir_preprocessed filesep dataset_name '_preprocessed'],'EEG_ICA','EEG_ICA_IClabelrejcomp','preprocess_segRej','bad_channels_removed');
%     elseif segment_data
%         save([curr_dir_preprocessed filesep dataset_name '_preprocessed'],'EEG_ICA','EEG_ICA_IClabelrejcomp','preprocess_segRej');
%     else
%         save([curr_dir_preprocessed filesep dataset_name '_preprocessed'],'EEG_ICA','EEG_ICA_IClabelrejcomp');
%     end
    if cfg.save_Workspace save([curr_dir_preprocessed filesep dataset_name '_preprocessed-Workspace']); end
    delete([curr_dir_preprocessed filesep 'Stage*.set']);
    delete([curr_dir_preprocessed filesep 'Stage*.fdt']);
    delete([curr_dir_preprocessed filesep '*Stage*-Workspace.mat']);
end