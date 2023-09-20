function EEG = add_events_from_latency_EEGfMRI(EEG,event_to_add_TYPE, event_to_add_LATENCY,event_to_add_DURATION,varargin)
% function EEG = add_events_from_latency_EEGfMRI(slice_marker,slice_EVENTS,EEG,event_to_add_TYPE, event_to_add_LATENCY,event_to_add_DURATION)
if ~isempty(varargin) % Only need this if the latencies are in terms of offset from first slice. If they are absolute latencies, do not provide varargin
    slice_marker = varargin{1};
    slice_EVENTS = varargin{2};
    
    first_slice_latency = find(cellfun(@(x) strcmp(x,slice_marker),{slice_EVENTS(:).type})); first_slice_latency = slice_EVENTS(first_slice_latency(1)).latency;
end

for kk = 1:length(event_to_add_LATENCY)
    for jj = 1:length(event_to_add_TYPE{kk})
        n_events = length(EEG.event);
        EEG.event(n_events+1).type = event_to_add_TYPE{kk}{jj};       
        if ~isempty(varargin) EEG.event(n_events+1).latency = first_slice_latency + event_to_add_LATENCY(kk);
        else EEG.event(n_events+1).latency = event_to_add_LATENCY(kk); end
        EEG.event(n_events+1).duration = event_to_add_DURATION(kk);
        EEG.event(n_events+1).urevent = n_events+1;
    end
end
EEG = eeg_checkset(EEG,'eventconsistency'); % Check for consistency and reorder the events chronologically


%% Temporary code:
% % First EEG_vhdr:
% EEG_vhdr{m} = add_events_from_latency_EEGfMRI(slice_marker,EEG_vhdr{m}.event,EEG_vhdr{m},study_conditions{k}, curr_latencies,curr_durations);
% 
% %                             first_slice_latency = find(cellfun(@(x) strcmp(x,slice_marker),{EEG_vhdr{m}.event(:).type})); first_slice_latency = EEG_vhdr{m}.event(first_slice_latency(1)).latency;
% %                             for kk = 1:length(curr_latencies)
% %                                 n_events = length(EEG_vhdr{m}.event);
% %                                 EEG_vhdr{m}.event(n_events+1).type = study_conditions{k};
% %                                 EEG_vhdr{m}.event(n_events+1).latency = first_slice_latency + curr_latencies(kk);
% %                                 EEG_vhdr{m}.event(n_events+1).duration = curr_durations(kk);
% %                                 EEG_vhdr{m}.event(n_events+1).urevent = n_events+1;
% %                             end
% %                             EEG_vhdr{m} = eeg_checkset(EEG_vhdr{m},'eventconsistency'); % Check for consistency and reorder the events chronologically
% %
% % Next EEG_mat:
% final_EEG_EVENT.code = final_EEG_EVENT.type; final_EEG_EVENT.type = final_EEG_EVENT.value;
% EEG_mat{m} = add_events_from_latency_EEGfMRI(slice_marker,final_EEG_EVENT,EEG_mat{m},study_conditions{k}, curr_latencies,curr_durations);
% 
% %                             first_slice_latency = find(cellfun(@(x) strcmp(x,slice_marker),{final_EEG_EVENT(:).value})); first_slice_latency = final_EEG_EVENT(first_slice_latency(1)).latency;
% %                             for kk = 1:length(curr_latencies)
% %                                 n_events = length(EEG_mat{m}.event);
% %                                 EEG_mat{m}.event(n_events+1).type = study_conditions{k};
% %                                 EEG_mat{m}.event(n_events+1).latency = first_slice_latency + curr_latencies(kk);
% %                                 EEG_vhdr{m}.event(n_events+1).duration = curr_durations(kk);
% %                                 EEG_mat{m}.event(n_events+1).urevent = n_events+1;
% %                             end
% %                             EEG_mat{m} = eeg_checkset(EEG_mat{m},'eventconsistency'); % Check for consistency and reorder the events chronologically
% 
% end