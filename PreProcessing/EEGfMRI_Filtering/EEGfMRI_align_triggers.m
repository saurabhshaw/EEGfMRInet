function [final_data] = EEGfMRI_align_triggers(correct_triggers,faulty_triggers,scan_parameters,final_data)
% Testing values:
% correct_triggers = EEG_vhdr{3}.event;
% faulty_triggers = EEG_mat{3}.event;
% scan_parameters;

% Extract slice triggers:
correct_slice_latencies = [correct_triggers(find(strcmp(scan_parameters.slice_marker,{correct_triggers.type}))).latency];
faulty_slice_latencies = [faulty_triggers(find(strcmp(scan_parameters.slice_marker,{faulty_triggers.type}))).latency];
correct_slice_idx = find(strcmp(scan_parameters.slice_marker,{correct_triggers.type}));
faulty_slice_idx = find(strcmp(scan_parameters.slice_marker,{faulty_triggers.type}));

% Compute with respect to first slice:
correct_slice_latencies_norm = correct_slice_latencies-correct_slice_latencies(1);
faulty_slice_latencies_norm = faulty_slice_latencies-faulty_slice_latencies(1);

% Compute diff:
correct_slice_latencies_diff = diff(correct_slice_latencies);
faulty_slice_latencies_diff = diff(faulty_slice_latencies);
timepnts_per_sliceevent = floor(nanmean(faulty_slice_latencies_diff));

% Add NaNs to bring the number of slice triggers to the same number:
length_diff = length(correct_slice_latencies_diff) - length(faulty_slice_latencies_diff);
if length_diff > 0
    correct_slice_latencies_mod = correct_slice_latencies;
    faulty_slice_latencies_mod = [faulty_slice_latencies nan(1,length_diff)];
    faulty_slice_latencies_norm = [faulty_slice_latencies_norm nan(1,length_diff)];
    faulty_slice_latencies_diff = [faulty_slice_latencies_diff nan(1,length_diff)];
elseif length_diff < 0
    faulty_slice_latencies_mod = faulty_slice_latencies;
    correct_slice_latencies_mod = [correct_slice_latencies nan(1,length_diff)];
    correct_slice_latencies_norm = [correct_slice_latencies_norm nan(1,abs(length_diff))];
    correct_slice_latencies_diff = [correct_slice_latencies_diff nan(1,abs(length_diff))];
end

% Find index of disparity between the two latencies:
latencies_error = (correct_slice_latencies_norm-faulty_slice_latencies_norm);
diff_error = diff(latencies_error);

% Positive deviation: correct_slice_latencies_norm > faulty_slice_latencies_norm:
latencies_error_pos_idx = find(diff_error>1); % This is > and not >= to avoid spurious deviations

% Negative deviation: correct_slice_latencies_norm < faulty_slice_latencies_norm:
latencies_error_neg_idx = find(((-1)*diff_error)>1); % This is > and not >= to avoid spurious deviations

% Pick the most relevant one:
if ~isempty(latencies_error_pos_idx)
    latencies_error_idx = latencies_error_pos_idx; 

elseif ~isempty(latencies_error_neg_idx)
    latencies_error_idx = latencies_error_neg_idx;
    
else
    latencies_error_idx = [];
end
latencies_error_vol_idx = floor(latencies_error_idx./scan_parameters.slicespervolume)+1;

if ~isempty(latencies_error_vol_idx)
    %% For each of the disparity, find the number of EEG datapoints missed:
    faulty_slice_latencies_missing_idx = []; faulty_slice_idx_fixed = []; faulty_slice_latencies_fixed = []; missed_vols = []; missed_slices = []; corrected_slice_markers = []; missed_slices_floor = [];
    for i = 1:length(latencies_error_vol_idx)
        curr_vol_idx = final_data.final_EEG_BLOCKS_dir_mod{latencies_error_vol_idx(i)};
        [curr_vol_delta,curr_vol_delta_idx] = max(diff(curr_vol_idx)-1);
        curr_missed_slices = double(curr_vol_delta)./(timepnts_per_sliceevent);
        curr_missed_vols = (curr_missed_slices/scan_parameters.slicespervolume);
        missed_vols = [missed_vols curr_missed_vols]; missed_slices = [missed_slices curr_missed_slices];
        
        % Accumulate the indices of slice markers that are currently missing:
        if ((curr_missed_slices < 1) && ~isempty(latencies_error_idx)) % Case when there is one missed slice, but the number of missed points are not equal to one full slice - i.e. just missed a few time points around the slice marker
            curr_missed_slices_floor = 1;
        else  % Case when number of missed points more than one full slice
            curr_missed_slices_floor = floor(curr_missed_slices);
        end
        missed_slices_floor = [missed_slices_floor curr_missed_slices_floor];
        faulty_slice_latencies_missing_idx = cat(2,faulty_slice_latencies_missing_idx,[latencies_error_idx(i):latencies_error_idx(i) + floor(curr_missed_slices)]);
        if i == 1 start_idx = 1; else start_idx = latencies_error_idx(i-1) + 1; end
        end_idx = latencies_error_idx(i);
        faulty_slice_latencies_fixed = cat(2,faulty_slice_latencies_fixed,[faulty_slice_latencies(start_idx:end_idx) nan(1,curr_missed_slices_floor)]); % end_idx = end_idx + floor(curr_missed_slices);
        faulty_slice_idx_fixed = cat(2,faulty_slice_idx_fixed,[faulty_slice_idx(start_idx:end_idx) nan(1,curr_missed_slices_floor)]);
        
%         faulty_slice_latencies_fixed = cat(2,faulty_slice_latencies_fixed,[faulty_slice_latencies(start_idx:end_idx) nan(1,floor(curr_missed_slices))]); % end_idx = end_idx + floor(curr_missed_slices);
%         faulty_slice_idx_fixed = cat(2,faulty_slice_idx_fixed,[faulty_slice_idx(start_idx:end_idx) nan(1,floor(curr_missed_slices))]);
%         
        %     % Accumulate the slice markers, by substituting the NaN with the markers from the correct_triggers
        %     corrected_slice_markers = cat(2,corrected_slice_markers,[final_data.final_EEG_EVENT(faulty_slice_idx(start_idx:end_idx))...
        %                                                             correct_triggers(correct_slice_idx(end_idx+1 : end_idx+floor(curr_missed_slices)))]);
    end
    start_idx = latencies_error_idx(end) + 1; end_idx = length(faulty_slice_latencies);
    faulty_slice_latencies_fixed = cat(2,faulty_slice_latencies_fixed,faulty_slice_latencies(start_idx:end_idx)) - faulty_slice_latencies_fixed(1);
    faulty_slice_idx_fixed = cat(2,faulty_slice_idx_fixed,faulty_slice_idx(start_idx:end_idx));
    
    % These last few slices did not add up to scan_parameters.slicespervolume and hence were not written out:
%     length_diff_fixed = length_diff - sum(floor(missed_slices));
    length_diff_fixed = length_diff - sum(missed_slices_floor);
    faulty_slice_latencies_fixed = cat(2,faulty_slice_latencies_fixed,nan(1,length_diff_fixed));
    faulty_slice_idx_fixed = cat(2,faulty_slice_idx_fixed,nan(1,length_diff_fixed));
    
    latencies_fixed_error = correct_slice_latencies_norm-(faulty_slice_latencies_fixed);
    % figure; plot(latencies_fixed_error)
    
    %% Remove the volumes that are partially acquired and are causing a frame-shift:
    % First create the correct order of slices expected:
    faulty_start_idx = 1:scan_parameters.slicespervolume:length(faulty_slice_idx_fixed); % faulty_end_idx = scan_parameters.slicespervolume:scan_parameters.slicespervolume:length(faulty_slice_idx_fixed);
    faulty_end_idx = faulty_start_idx + scan_parameters.slicespervolume - 1;
    faulty_idx_outofbounds = (faulty_end_idx > length(faulty_slice_idx_fixed));
    faulty_start_idx = faulty_start_idx(~faulty_idx_outofbounds); faulty_end_idx = faulty_end_idx(~faulty_idx_outofbounds);
    
    % Remove the volumes that are partially acquired and hence corrupted:
    faulty_vol_idx_fixed_start = faulty_slice_idx_fixed(faulty_start_idx); faulty_vol_idx_fixed_end = faulty_slice_idx_fixed(faulty_end_idx); faulty_vol_label_fixed = 1:length(faulty_vol_idx_fixed_start);
    faulty_vol_idx_fixed = arrayfun(@(x,y)faulty_slice_idx_fixed(x:y),faulty_start_idx,faulty_end_idx,'un',0);
    faulty_vol_idx_toremove = cellfun(@(x)sum(isnan(x)),faulty_vol_idx_fixed);
    final_idx_cell = faulty_vol_idx_fixed(faulty_vol_idx_toremove == 0);
    final_idx = cell2mat(final_idx_cell); final_idx_label = faulty_vol_label_fixed(faulty_vol_idx_toremove == 0);
    removed_idx_cell = faulty_vol_idx_fixed(faulty_vol_idx_toremove > 0);
    removed_idx = cell2mat(removed_idx_cell); removed_idx = removed_idx(~isnan(removed_idx));
    
    faulty_triggers_corrected = faulty_triggers(final_idx);
    
    %% Find the correspondence of the initial faulty volume numbers and the correct volume numbers (frame-shift error):
    faulty_start_idx = 1:scan_parameters.slicespervolume:length(faulty_slice_idx); %faulty_end_idx = scan_parameters.slicespervolume:scan_parameters.slicespervolume:length(faulty_slice_idx);
    faulty_end_idx = faulty_start_idx + scan_parameters.slicespervolume - 1;
    faulty_idx_outofbounds = (faulty_end_idx > length(faulty_slice_idx_fixed));
    faulty_start_idx = faulty_start_idx(~faulty_idx_outofbounds); faulty_end_idx = faulty_end_idx(~faulty_idx_outofbounds);
    
    faulty_vol_idx_start = faulty_slice_idx(faulty_start_idx); faulty_vol_idx_end = faulty_slice_idx(faulty_end_idx); faulty_vol_label = 1:length(faulty_vol_idx_start);
    faulty_vol_idx = arrayfun(@(x,y)faulty_slice_idx(x:y),faulty_start_idx,faulty_end_idx,'un',0);
    
    % faulty_fixed_idx_intersect = cellfun(@(x) cellfun(@(y)intersect(x,y),faulty_vol_idx_fixed,'un',0) ,faulty_vol_idx,'un',0);
    faulty_fixed_idx_intersect = cellfun(@(x) find(cellfun(@(y)~isempty(intersect(x,y)),faulty_vol_idx_fixed)) ,faulty_vol_idx,'un',0);
    % faulty_vol_label_trans = faulty_vol_label_fixed(~isnan(faulty_vol_idx_fixed_end));
    
    %% Add "boundary" events at the start of the removed incomplete volumes:
    removed_start_idx = cellfun(@(x)x(1),removed_idx_cell); removed_start_idx = removed_start_idx(~isnan(removed_start_idx));
    removed_start_latencies = [faulty_triggers(removed_start_idx).latency];
    removed_start_latencies = repmat(removed_start_latencies',[1 2]);
    % [EEG.event] = eeg_insertbound(EEG.event, EEG.pnts, removed_start_latencies);
    % EEG = eeg_checkset(EEG, 'eventconsistency');
    
    %% Create a fixed final_EEG_EVENT with the events filled in from the correct events to properly identify the block onsets later:
    idx_linear_diff = diff(isnan(faulty_slice_idx_fixed));
    correct_start_idx = find(idx_linear_diff > 0);
    correct_end_idx = find(idx_linear_diff < 0);
    if (length(correct_end_idx) < length(correct_start_idx)) correct_end_idx = [correct_end_idx length(faulty_slice_idx_fixed)];end
    
    final_EEG_RDAEVENT_fixed = []; RDA_blocksize = final_data.final_EEG_EVENT(1).RDAblocksize;
    for i = 1:length(correct_start_idx)
        
        % Get indices of Faulty triggers to be appended to final_EEG_RDAEVENT_fixed
        if i == 1 curr_faulty_start_idxIDX = 1; else curr_faulty_start_idxIDX = correct_end_idx(i-1) + 1; end
        curr_faulty_end_idxIDX = correct_start_idx(i);
        curr_faulty_idx = faulty_slice_idx_fixed(curr_faulty_start_idxIDX:curr_faulty_end_idxIDX);
        
        % Get indices of Correct triggers to be appended to final_EEG_RDAEVENT_fixed
        curr_correct_start_idx = correct_start_idx(i); curr_correct_end_idx = correct_end_idx(i);
        curr_correct_idx = correct_slice_idx(curr_correct_start_idx:curr_correct_end_idx);
        
        % Append the selected Faulty triggers
        selected_slice_events = final_data.final_EEG_EVENT(curr_faulty_idx);
        curr_RDAEVENTS = arrayfun(@(x)((x.RDAblock)*RDA_blocksize+x.RDAposition),selected_slice_events);
        final_EEG_RDAEVENT_fixed = cat(2,final_EEG_RDAEVENT_fixed,curr_RDAEVENTS);
        
        % Append the selected Correct triggers
        curr_correct_block = [correct_triggers(curr_correct_idx).latency];
        curr_correct_RDAEVENTS = arrayfun(@(x)x-curr_correct_block(1)+curr_RDAEVENTS(end),curr_correct_block);
        final_EEG_RDAEVENT_fixed = cat(2,final_EEG_RDAEVENT_fixed,curr_correct_RDAEVENTS(2:end)); % Remove the first item that is purposely included earlier to compute latencies with respect to the last slice marker
    end
    
    % final_EEG_RDAEVENT_fixed = final_data.final_EEG_EVENT(faulty_slice_idx_fixed);
    % final_EEG_EVENT_fixed(isnan(faulty_vol_idx_fixed_linear)) = correct_triggers(isnan(faulty_vol_idx_fixed_linear));
    % final_EEG_RDAEVENT_fixed(~isnan(faulty_slice_idx_fixed)) = final_data.final_EEG_EVENT(~isnan(faulty_slice_idx_fixed));
else
    
    faulty_triggers_corrected = [];
    final_idx = [];
    final_idx_label = [];
    faulty_fixed_idx_intersect = [];
    removed_start_latencies = [];
    final_idx_cell = [];
    final_EEG_RDAEVENT_fixed = [];    
    
end
%% Assign output variables:
final_data.faulty_triggers_corrected = faulty_triggers_corrected;
final_data.final_idx = final_idx;
final_data.final_idx_label = final_idx_label;
final_data.faulty_fixed_idx_intersect = faulty_fixed_idx_intersect;
final_data.removed_start_latencies = removed_start_latencies;
final_data.final_idx_cell = final_idx_cell;
final_data.final_EEG_RDAEVENT_fixed = final_EEG_RDAEVENT_fixed;