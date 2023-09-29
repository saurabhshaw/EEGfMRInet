function [EEG, start_idx, end_idx] = create_windows(EEG, scan_param, feature_param)

% create epochs over which to compute features - sliding windows
slice_latencies = floor([EEG.event(find(strcmp(num2str(scan_param.slice_marker),{EEG.event.type}))).latency]);
start_idx = min(slice_latencies);
max_idx = max(slice_latencies);
append_idx = start_idx;
window_step = feature_param.window_step*EEG.srate;
window_length = feature_param.window_length*EEG.srate;
while (start_idx(end) < max_idx)
    start_idx = [start_idx, append_idx+window_step];
    append_idx = start_idx(end);
end
end_idx = ceil(start_idx + window_length)-1;

temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});

