function EEG = add_MRItrial_markers_EEGfMRI(EEG, curr_tD_file, scan_parameters ,m)
%% Add individual trial markers (101, 201, etc..)
% Find the onset and duration of each trial in terms of the MRI volumes
curr_class_markers_MRIonset = cell(1,length(curr_tD_file.class_MARKERS_BLOCKIDX_vect{m}));
curr_class_markers_MRIduration = cell(1,length(curr_tD_file.class_MARKERS_BLOCKIDX_vect{m}));
for idx_i = 1:length(curr_tD_file.class_MARKERS_BLOCKIDX_vect{m})
    x = curr_tD_file.class_MARKERS_BLOCKIDX_vect{m}{idx_i};
    for idx_j = 1:size(x,2)
        curr_class_markers_MRIonset{idx_i}{idx_j} = find(diff(curr_tD_file.MRI_start_BLOCKIDX_vect{m} < x(1,idx_j))) + 1;
        curr_class_markers_MRIduration{idx_i}{idx_j} = find(diff(curr_tD_file.MRI_end_BLOCKIDX_vect{m} > x(2,idx_j))) + 1;
    end
    
    % Replace empty entries with zero in curr_class_markers_MRI_onset
    temp_idx = find(cell2mat(cellfun(@(x)isempty(x),curr_class_markers_MRIonset{idx_i},'UniformOutput',0)));
    if ~isempty(temp_idx)
        for mm = 1:length(temp_idx) curr_temp_idx = temp_idx(mm); curr_class_markers_MRIonset{idx_i}{curr_temp_idx} = 0; end
    end
    curr_class_markers_MRIonset{idx_i} = cell2mat(curr_class_markers_MRIonset{idx_i});
    
    % Replace empty entries with 1 in curr_class_markers_MRI_duration
    temp_idx = find(cell2mat(cellfun(@(x)isempty(x),curr_class_markers_MRIduration{idx_i},'UniformOutput',0)));
    if ~isempty(temp_idx)
        for mm = 1:length(temp_idx) curr_temp_idx = temp_idx(mm); curr_class_markers_MRIduration{idx_i}{curr_temp_idx} = 1; end
    end
    curr_class_markers_MRIduration{idx_i} = cell2mat(curr_class_markers_MRIduration{idx_i})-curr_class_markers_MRIonset{idx_i};
end

% Add to EEG:
EEG_SLICE_latency = find(cellfun(@(x) strcmp(x,scan_parameters.slice_marker),{EEG.event(:).type})); EEG_SLICE_latency = cell2mat({EEG.event(EEG_SLICE_latency).latency});
EEG_VOLUME_latency = EEG_SLICE_latency(1:scan_parameters.slicespervolume:length(EEG_SLICE_latency));
for k = 1:length(curr_class_markers_MRIonset)
    curr_MRI_vol = curr_class_markers_MRIonset{k}; curr_MRI_withinbounds = (curr_MRI_vol>0) & (curr_MRI_vol<=length(EEG_VOLUME_latency));
    curr_MRI_vol = curr_MRI_vol(curr_MRI_withinbounds); % Remove the markers that are before the onset of the MRI scanning
    curr_latency = EEG_VOLUME_latency(curr_MRI_vol); % curr_latency = curr_latency(curr_MRI_vol>0);
    curr_duration = curr_class_markers_MRIduration{k}(curr_MRI_withinbounds).*scan_parameters.srate; % curr_duration = curr_duration(curr_MRI_vol>0);
    curr_type = mat2cell(curr_tD_file.class_MARKERS_vect{m}{k}(curr_MRI_withinbounds),1,ones(1,length(curr_duration))); % curr_type = curr_type(curr_MRI_vol>0);
    curr_type = cellfun(@(x){['MRI_' num2str(x)]},curr_type,'un',0);
    
    % Add EEG:
    EEG = add_events_from_latency_EEGfMRI(EEG,curr_type, curr_latency,curr_duration);
    
    %% Moved within add_events_from_latency_EEGfMRI %%
%     for kk = 1:length(curr_class_markers_MRIonset{k})
%         curr_MRI_vol = curr_class_markers_MRIonset{k}(kk);
%         if curr_MRI_vol > 0
%             n_events = length(EEG_vhdr{m}.event);
%             EEG_vhdr{m}.event(n_events+1).type = curr_tD_file.class_MARKERS_vect{m}{k}(kk);
%             EEG_vhdr{m}.event(n_events+1).latency = EEG_vhdr_VOLUME_latency(curr_MRI_vol);
%             EEG_vhdr{m}.event(n_events+1).duration = curr_class_markers_MRIduration{k}(kk)*srate;
%             EEG_vhdr{m}.event(n_events+1).urevent = n_events+1;
%         end
%     end
end