function [EEG] = eegfmri_create_windows(EEG, scan_param)

start_idx = ceil([EEG.event(find(strcmp(num2str(scan_param.slice_marker),{EEG.event.type}))).latency]);
end_idx = ceil(start_idx + scan_param.TR*EEG.srate)-1;

temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});