function EEG = add_MRIcondition_markers_EEGfMRI(EEG, curr_tD_file, scan_parameters ,m)
%% Add condition markers to both EEG streams:
% Based on MRIonset and MRIduration as defined for the CONN processing:
curr_condition_MRIonset = cellfun(@(x)x*scan_parameters.srate,curr_tD_file.block_condition_onset_vect{m},'un',0);
curr_condition_MRIduration = cellfun(@(x)x*scan_parameters.srate,curr_tD_file.block_condition_duration_vect{m},'un',0);
for k = 1:length(curr_condition_MRIonset)
    curr_conditions = cellfun(@(x) {['MRI_' curr_tD_file.study_conditions{k}]},cell(1,length(curr_condition_MRIonset{k})),'un',0); % Make sure the names are in individual cells of their own
    
    % Add to EEG:
    EEG = add_events_from_latency_EEGfMRI(EEG,curr_conditions, curr_condition_MRIonset{k}, curr_condition_MRIduration{k},scan_parameters.slice_marker,EEG.event);
    
end

%% Old code:
                        
% curr_condition_MRIonset = cellfun(@(x)x*srate,curr_tD_file.block_condition_onset_vect{m},'un',0);
% curr_condition_MRIduration = cellfun(@(x)x*srate,curr_tD_file.block_condition_duration_vect{m},'un',0);
% for k = 1:length(curr_condition_MRIonset)
%     curr_conditions = cellfun(@(x) {['MRI_' study_conditions{k}]},cell(1,length(curr_condition_MRIonset{k})),'un',0); % Make sure the names are in individual cells of their own
%     
%     % Add to EEG_vhdr:
%     EEG_vhdr{m} = add_events_from_latency_EEGfMRI(EEG_vhdr{m},curr_conditions, curr_condition_MRIonset{k}, curr_condition_MRIduration{k},scan_parameters.slice_marker,EEG_vhdr{m}.event);
%     
%     % Add to EEG_mat:
%     EEG_mat{m} = add_events_from_latency_EEGfMRI(EEG_mat{m},curr_conditions, curr_condition_MRIonset{k}, curr_condition_MRIduration{k},scan_parameters.slice_marker,EEG_mat{m}.event);
% end
                        