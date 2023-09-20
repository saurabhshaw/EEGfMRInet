function [EEG,final_data] = curate_mat2set_EEGfMRI(final_data,chanlocs_file,dataset_name,scan_parameters,varargin)

final_EEG = final_data.final_EEG;
final_EEG_EVENT = final_data.final_EEG_EVENT;
final_EEG_BLOCKS_dir = final_data.final_EEG_BLOCKS_dir;
final_EEG_EVENT_dir_mod = final_data.final_EEG_EVENT_dir_mod;
final_EEG_EVENT_LATENCY_dir = final_data.final_EEG_EVENT_LATENCY_dir;

% Import into EEGLAB format:
% EEG = pop_importdata('dataformat','array','nbchan',0,'data','final_EEG','srate',scan_parameters.srate,'pnts',0,'xmin',0);
EEG = pop_importdata('dataformat','array','nbchan',0,'data',final_EEG,'srate',scan_parameters.srate,'pnts',0,'xmin',0);
EEG = add_events_from_latency_EEGfMRI(EEG,{final_EEG_EVENT(:).type}, cell2mat({final_EEG_EVENT(:).latency}),cell2mat({final_EEG_EVENT(:).duration}));

EEG = pop_chanedit(EEG,'load',{chanlocs_file 'filetype' 'autodetect'});
EEG.setname = [dataset_name '_MAT']; EEG = eeg_checkset( EEG );


% The case when the scan was ended prematurely:
if length(find(strcmp(scan_parameters.slice_marker,{EEG.event(:).type}))) < 0.75*scan_parameters.tfunc_num_images
    
    % Fix this!!!
    correct_triggers = varargin{1}; [final_data] = EEGfMRI_align_triggers(correct_triggers,EEG.event,scan_parameters,final_data);
    if ~isempty(final_data.faulty_triggers_corrected) EEG.event = final_data.faulty_triggers_corrected; end
    
    % Add "boundary" events at the start of the removed incomplete volumes:
    if ~isempty(final_data.removed_start_latencies) [EEG.event] = eeg_insertbound(EEG.event, EEG.pnts, final_data.removed_start_latencies); end
    EEG = eeg_checkset(EEG, 'eventconsistency');
    
elseif length(find(strcmp(scan_parameters.slice_marker,{EEG.event(:).type}))) < scan_parameters.tfunc_num_images
      
    if ~isempty(varargin) % To detect spurious discontinuities (signal interruption) in EEG_MAT markers by comparing to EEG_VHDR
        correct_triggers = varargin{1}; [final_data] = EEGfMRI_align_triggers(correct_triggers,EEG.event,scan_parameters,final_data);
        if ~isempty(final_data.faulty_triggers_corrected) EEG.event = final_data.faulty_triggers_corrected; end
        
        % Add "boundary" events at the start of the removed incomplete volumes:
        if ~isempty(final_data.removed_start_latencies) [EEG.event] = eeg_insertbound(EEG.event, EEG.pnts, final_data.removed_start_latencies); end
        EEG = eeg_checkset(EEG, 'eventconsistency');
        
    else
        
        % Detect discontinuities based on difference in EEG BLOCKS length and insert "boundary" events:
        total_num_vols = length(final_EEG_BLOCKS_dir); pnts_per_vol = scan_parameters.TR*scan_parameters.srate;
        data_breaks = find(cellfun(@(x) length(x),final_EEG_BLOCKS_dir) < pnts_per_vol)'; data_breaks_idx = zeros(length(data_breaks),2);
        for i = 1:length(data_breaks)
            curr_i = find(arrayfun(@(x)strcmp(x,scan_parameters.slice_marker),{final_EEG_EVENT_dir_mod{data_breaks(i)}.value}),1,'first');
            data_breaks_idx(i,:) = [final_EEG_EVENT_LATENCY_dir{data_breaks(i)}(curr_i) final_EEG_EVENT_LATENCY_dir{data_breaks(i)}(curr_i)];
        end
        [EEG.event] = eeg_insertbound(EEG.event, EEG.pnts, data_breaks_idx);
        EEG = eeg_checkset(EEG, 'eventconsistency');
    end
end


