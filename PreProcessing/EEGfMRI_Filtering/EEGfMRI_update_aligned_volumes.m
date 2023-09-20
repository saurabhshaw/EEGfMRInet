function curr_tD_file = EEGfMRI_update_aligned_volumes(curr_tD_file,final_data,scan_parameters,within_addSubjects,m)

if size(final_data.removed_start_latencies,1) >= 1
    % Backup current values of fields that will be updated:
    if ~within_addSubjects % dont do this if this function is being called from inside conn_addSubjects_EEGfMRI:
        curr_tD_file.ORIG_block_onset_vect = curr_tD_file.block_onset_vect;
        curr_tD_file.ORIG_block_duration_vect = curr_tD_file.block_duration_vect;
        curr_tD_file.ORIG_block_condition_onset_vect = curr_tD_file.block_condition_onset_vect;
        curr_tD_file.ORIG_block_condition_duration_vect = curr_tD_file.block_condition_duration_vect;
    end
    curr_tD_file.ORIG_MRI_start_BLOCKIDX_vect = curr_tD_file.MRI_start_BLOCKIDX_vect;
    curr_tD_file.ORIG_MRI_end_BLOCKIDX_vect = curr_tD_file.MRI_end_BLOCKIDX_vect;
    
    %% Isolate the start and end of each RDA block from the corrected final_data.final_EEG_RDAEVENT_fixed
    RDA_blocksize = double(final_data.final_EEG_EVENT(1).RDAblocksize);
    start_idx = 1:scan_parameters.slicespervolume:length(final_data.final_EEG_RDAEVENT_fixed); 
    % end_idx = scan_parameters.slicespervolume:scan_parameters.slicespervolume:length(final_data.final_EEG_RDAEVENT_fixed);
    end_idx = start_idx + scan_parameters.slicespervolume - 1;
    idx_outofbounds = (end_idx > length(final_data.final_EEG_RDAEVENT_fixed));
    start_idx = start_idx(~idx_outofbounds); end_idx = end_idx(~idx_outofbounds);
    
    selected_volume_events_RDAidx = arrayfun(@(x,y)final_data.final_EEG_RDAEVENT_fixed(x:y),start_idx,end_idx,'un',0);
    selected_volume_RDABLOCKidx = cellfun(@(x) floor(double([x(1) x(end)])./RDA_blocksize),selected_volume_events_RDAidx,'un',0);

    curr_tD_file.MRI_start_BLOCKIDX_vect{m} = cellfun(@(x) x(1),selected_volume_RDABLOCKidx);
    curr_tD_file.MRI_end_BLOCKIDX_vect{m} = cellfun(@(x) x(2),selected_volume_RDABLOCKidx);

    %% Re-compute block_onset_vect and block_duration_vect (as done in conn_addSubjects_EEGfMRI):
    if ~within_addSubjects % dont do this if this function is being called from inside conn_addSubjects_EEGfMRI:
        curr_tD_file.block_onset_vect{m} = cellfun(@(x)find(diff(curr_tD_file.MRI_start_BLOCKIDX_vect{m} < x(1))) + 1,curr_tD_file.block_BLOCKIDX_vect{m},'UniformOutput',0);
        temp_idx = find(cell2mat(cellfun(@(x)isempty(x),curr_tD_file.block_onset_vect{m},'UniformOutput',0)));
        if ~isempty(temp_idx)
            for kk = 1:length(temp_idx) curr_temp_idx = temp_idx(kk); curr_tD_file.block_onset_vect{m}{curr_temp_idx} = 0; end
        end
        curr_tD_file.block_onset_vect{m} = cell2mat(curr_tD_file.block_onset_vect{m})*curr_tD_file.TR; % Multiply by TR to convert to seconds
        
        curr_tD_file.block_duration_vect{m} = cellfun(@(x)find(diff(curr_tD_file.MRI_end_BLOCKIDX_vect{m} > x(2))) + 1,curr_tD_file.block_BLOCKIDX_vect{m},'UniformOutput',0);
        temp_idx = find(cell2mat(cellfun(@(x)isempty(x),curr_tD_file.block_duration_vect{m},'UniformOutput',0)));
        if ~isempty(temp_idx)
            for kk = 1:length(temp_idx) curr_temp_idx = temp_idx(kk); curr_tD_file.block_duration_vect{m}{curr_temp_idx} = 0; end
        end
        curr_tD_file.block_duration_vect{m} = cell2mat(curr_tD_file.block_duration_vect{m})*curr_tD_file.TR - curr_tD_file.block_onset_vect{m}; % Multiply by TR to convert to seconds
        
        
        %% Re-compute block_condition_onset_vect and block_condition_duration_vect (as done in conn_addSubjects_EEGfMRI):
        curr_tD_file.block_condition_onset_vect{m} = cell(1,length(curr_tD_file.study_conditions));
        curr_tD_file.block_condition_duration_vect{m} = cell(1,length(curr_tD_file.study_conditions));
        for k = 1:length(curr_tD_file.study_conditions)
            curr_condition_vect = curr_tD_file.block_condition_vect{m}(k,:); min_val = min(curr_condition_vect(curr_condition_vect~=0));
            
            if (k >= 4) && (~isempty(min_val))
                min_condition_vect = find(curr_condition_vect==min_val);
                curr_tD_file.block_condition_onset_vect{m}{k} = curr_tD_file.block_onset_vect{m}(min_condition_vect);
                curr_tD_file.block_condition_duration_vect{m}{k} = curr_tD_file.block_duration_vect{m}(min_condition_vect) + curr_tD_file.block_duration_vect{m}(min_condition_vect + 1);
            else
                curr_tD_file.block_condition_onset_vect{m}{k} = curr_tD_file.block_onset_vect{m}(curr_tD_file.block_condition_vect{m}(k,:) > 0);
                curr_tD_file.block_condition_duration_vect{m}{k} = curr_tD_file.block_duration_vect{m}(curr_tD_file.block_condition_vect{m}(k,:) > 0);
            end
        end
    
    %% Write this out to the Trial_data file:
%     subject_folder = curr_tD_file.subject_folder_vect{curr_tD_file.i};
%     subject_dir = [curr_tD_file.base_path_data filesep subject_folder];
%     curr_ID = curr_tD_file.subjectID_vect{curr_tD_file.i};
%     save([subject_dir filesep curr_ID '_trial_data'],'-struct','curr_tD_file');
    end
else
    curr_tD_file.ORIG_MRI_start_BLOCKIDX_vect = cell(size(curr_tD_file.MRI_start_BLOCKIDX_vect));
    curr_tD_file.ORIG_MRI_end_BLOCKIDX_vect = cell(size(curr_tD_file.MRI_end_BLOCKIDX_vect));   
end
