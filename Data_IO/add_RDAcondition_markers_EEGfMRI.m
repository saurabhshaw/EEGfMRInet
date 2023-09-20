function EEG = add_RDAcondition_markers_EEGfMRI(EEG, curr_tD_file, scan_parameters, RDA_blocksize,m)
%% Add condition markers to EEG stream Based on RDA blocks of BLOCKIDX 
%(The "curr_onset" is with respect to the first slice marker - curr_tD_file.MRI_start_BLOCKIDX_vect{m}(1)):

first_slice_onset_RDAblock = curr_tD_file.MRI_start_BLOCKIDX_vect{m}(1);
curr_onset = (cellfun(@(x)x(1),curr_tD_file.block_BLOCKIDX_vect{m}) - first_slice_onset_RDAblock)*RDA_blocksize;
curr_duration = cellfun(@(x)x(2)-x(1),curr_tD_file.block_BLOCKIDX_vect{m})*RDA_blocksize;

curr_block_condition_vect = curr_tD_file.block_condition_vect{m};
curr_conditions_idx = cell(1,size(curr_block_condition_vect,2)); curr_conditions = cell(1,size(curr_block_condition_vect,2));
for k = 1:size(curr_block_condition_vect,2)
    curr_conditions_idx{k} = find(curr_block_condition_vect(:,k)>0);
    curr_conditions{k} = [curr_conditions{k} curr_tD_file.study_conditions(curr_conditions_idx{k})];
end

% Add to EEG:
EEG = add_events_from_latency_EEGfMRI(EEG,curr_conditions, curr_onset,curr_duration,scan_parameters.slice_marker,EEG.event);

%% Old Code:
                        
% first_slice_onset_RDAblock = curr_tD_file.MRI_start_BLOCKIDX_vect{m}(1);
% curr_onset = (cellfun(@(x)x(1),curr_tD_file.block_BLOCKIDX_vect{m}) - first_slice_onset_RDAblock)*RDA_blocksize;
% curr_duration = cellfun(@(x)x(2)-x(1),curr_tD_file.block_BLOCKIDX_vect{m})*RDA_blocksize;
% 
% curr_block_condition_vect = curr_tD_file.block_condition_vect{m};
% curr_conditions_idx = cell(1,size(curr_block_condition_vect,2)); curr_conditions = cell(1,size(curr_block_condition_vect,2));
% for k = 1:size(curr_block_condition_vect,2)
%     curr_conditions_idx{k} = find(curr_block_condition_vect(:,k)>0);
%     curr_conditions{k} = [curr_conditions{k} study_conditions(curr_conditions_idx{k})];
% end
% 
% % Add to EEG_vhdr:
% EEG_vhdr{m} = add_events_from_latency_EEGfMRI(EEG_vhdr{m},curr_conditions, curr_onset,curr_duration,scan_parameters.slice_marker,EEG_vhdr{m}.event);
% 
% % Add to EEG_mat:
% EEG_mat{m} = add_events_from_latency_EEGfMRI(EEG_mat{m},curr_conditions, curr_onset,curr_duration,scan_parameters.slice_marker,EEG_mat{m}.event);
