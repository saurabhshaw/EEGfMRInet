function [EEG] = remove_extra_slicemarkers_EEGfMRI(EEG,scan_parameters,num_images)
% Use this function to identify and remove any extra slice markers that
% might be present before the scanning block begins due to fMRI
% pre-scanning procedures

delta_latency = 10; % Was 20
expected_latencydiff = (scan_parameters.TR/scan_parameters.slicespervolume)*EEG.srate;

curr_num_slice_markers = cellfun(@(x)strcmp(x,scan_parameters.slice_marker),{EEG.event(:).type}); 
if sum(curr_num_slice_markers) > num_images
    curr_slice_markers_latencydiff = diff(cell2mat({EEG.event(curr_num_slice_markers).latency}));    
    curr_slice_markers_outbounds_last = find(curr_slice_markers_latencydiff > (expected_latencydiff + delta_latency),1,'last');    
    if isempty(curr_slice_markers_outbounds_last) curr_slice_markers_outbounds_last = 0; end
    curr_final_idx = cell2mat({EEG.event(curr_num_slice_markers).urevent}); curr_final_idx = curr_final_idx(curr_slice_markers_outbounds_last+1);
    EEG.event = EEG.event(curr_final_idx:length(curr_num_slice_markers));
    EEG = eeg_checkset(EEG,'eventconsistency');
end

% Remove last triggers to compensate for any stray slice triggers - NEED TO FIND A BETTER WAY OF DOING THIS
curr_num_slice_markers = cellfun(@(x)strcmp(x,scan_parameters.slice_marker),{EEG.event(:).type}); 
if sum(curr_num_slice_markers) > num_images
    curr_diff = sum(curr_num_slice_markers) - num_images;
    curr_num_slice_markers_idx = find(curr_num_slice_markers);
    slices_markers_toremove = curr_num_slice_markers_idx((end-curr_diff+1):end);
    EEG.event(slices_markers_toremove) = [];
    EEG = eeg_checkset(EEG,'eventconsistency');    
end