
%% Control parameters:
filter_lp = 0.1; % Was 1 Hz
filter_hp = 50; % Was 40 Hz
segment_data = 0; % 1 for epoched data, 0 for continuous data
segment_markers = {}; % {} is all
task_segment_start = -0.5; % start of the segments in relation to the marker
task_segment_end = 5; % end of the segments in relation to the marker

%% Make folder if not already made:
if ~isdir ([curr_dir filesep 'PreProcessed'])
    mkdir ([curr_dir filesep 'PreProcessed']);
end
curr_dir_preprocessed = [curr_dir filesep 'PreProcessed'];

%% Filter the data:
% EEG = pop_eegfiltnew(EEG,filter_lp,filter_hp,8448,0,[],1); EEG.setname = [EEG.setname '_filt']; EEG = eeg_checkset( EEG );
EEG = pop_eegfiltnew(EEG,filter_lp,filter_hp); EEG.setname = [EEG.setname '_filt']; EEG = eeg_checkset( EEG );

%% Segment data according to data type
if segment_data
    EEG = pop_epoch(EEG, segment_markers, [task_segment_start task_segment_end], 'epochinfo', 'yes');
    EEG.setname = [EEG.setname '_segmented'];
    EEG = pop_saveset( EEG, 'filename',[EEG.setname '.set'],'filepath',curr_dir_preprocessed);
end

%% Detect bad channels:
% Using spectrum criteria and 3SDeviations as channel outlier threshold, done twice
full_channel_locations = EEG.chanlocs;
EEG = pop_rejchan(EEG, 'elec',[1:size(EEG.data,1)],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
EEG = eeg_checkset( EEG );

EEG = pop_rejchan(EEG, 'elec',[1:EEG.nbchan],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
EEG = eeg_checkset( EEG );
selected_channel_locations = EEG.chanlocs; selected_channel_labels = {selected_channel_locations.labels};
bad_channels_removed = setdiff({full_channel_locations(:).labels}, selected_channel_labels);

%% Run ICA and reject components that are not Brain or Other (according to ICLabel):
EEG = pop_runica(EEG, 'extended',1,'interupt','on'); EEG = eeg_checkset( EEG );
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
EEG = pop_subcomp(EEG, components, 0);
EEG.setname = [EEG.setname '_ICAreject']; EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename',[EEG.setname '.set'],'filepath',curr_dir_preprocessed);
% eeglab redraw
% pop_selectcomps(EEG, [1:10] );EEG = eeg_checkset( EEG );

%% ReReference the data:
EEG = pop_reref( EEG, []); EEG.setname = [EEG.setname '_reRef']; EEG = eeg_checkset( EEG );

%% Reject segments:
if segment_data
    reject_min_amp = -40; reject_max_amp = 40; pipeline_visualizations_semiautomated = 0;
    % EEG = pop_eegthresh(EEG,1,[1:EEG.nbchan] ,[reject_min_amp],[reject_max_amp],[EEG.xmin],[EEG.xmax],2,0);
    EEG = pop_jointprob(EEG,1,[1:EEG.nbchan],3,3,pipeline_visualizations_semiautomated,...
        0,pipeline_visualizations_semiautomated);
    
    EEG = eeg_rejsuperpose(EEG, 1, 0, 1, 1, 1, 1, 1, 1);
    EEG = pop_rejepoch(EEG, [EEG.reject.rejglobal] ,0);
    EEG.setname = [EEG.setname '_segRej']; EEG = eeg_checkset( EEG );
    EEG = pop_saveset(EEG, 'filename',[EEG.setname '.set'],'filepath',curr_dir_preprocessed);
end

%% Interpolate bad channels identified earlier:
EEG = pop_interp(EEG, full_channel_locations, 'spherical');
EEG.setname = [EEG.setname '_chanInterp']; EEG = eeg_checkset( EEG );
EEG = pop_saveset(EEG, 'filename',[EEG.setname '.set'],'filepath',curr_dir_preprocessed);

%% Save final pre-processed dataset:
EEG = pop_saveset(EEG, 'filename',[dataset_name '_preprocessed.set'],'filepath',curr_dir_preprocessed);