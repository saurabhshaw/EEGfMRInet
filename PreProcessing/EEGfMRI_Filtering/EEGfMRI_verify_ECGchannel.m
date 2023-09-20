function [EEG] = EEGfMRI_verify_ECGchannel(EEG,scan_parameters)

test_width = 10*EEG.srate; test_start_delta = 5*EEG.srate;

 % Identify the start of the triggers:
 Slice_marker_latencies = [EEG.event(find(strcmp(scan_parameters.slice_marker,{EEG.event.type}))).latency];
 startidx = max(min(Slice_marker_latencies) - test_width,1);
 % startidx = min(Slice_marker_latencies) + test_start_delta;
 endidx = startidx + test_width;
 
 test_data = rms(EEG.data(:,startidx:endidx)');
 [~,estimated_ECGchannel] = max(test_data); 
 
 if (mod(estimated_ECGchannel,size(EEG.data,1)/2) > 0) test_data(estimated_ECGchannel) = 0; [~,estimated_ECGchannel] = max(test_data); end
 
 if estimated_ECGchannel > scan_parameters.ECG_channel
     fprintf(['\n ***************************** Correcting ECG Channel ***************************** \n']);

     temp_data = EEG.data;
     EEG.data(1:scan_parameters.ECG_channel,:) = temp_data((scan_parameters.ECG_channel+1):size(EEG.data,1),:);
     EEG.data((scan_parameters.ECG_channel+1):size(EEG.data,1),:) = temp_data(1:scan_parameters.ECG_channel,:);
 end