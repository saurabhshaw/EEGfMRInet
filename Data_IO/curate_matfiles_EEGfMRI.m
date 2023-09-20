function [final_data] = curate_matfiles_EEGfMRI(curr_dir,study_prefix,scan_parameters)

srate = scan_parameters.srate;

curr_block_dir = dir([curr_dir filesep study_prefix 'data_volume*.mat']);
temp_reorderIDX = cellfun(@(x) strsplit(x,{'volume','_'}),{curr_block_dir.name},'un',0); % Reorder the read in files according to numerical order (dir loads it in alphabetical order)
[~,reorderIDX] = sort(cellfun(@(x) str2num(x{4}),temp_reorderIDX));
curr_block_dir = curr_block_dir(reorderIDX);
curr_block_dir_loaded = cellfun(@(x,y) load([x filesep y]),{curr_block_dir.folder},{curr_block_dir.name},'un',0);

% Concatenate data into one vector:
final_EEG_dir = cellfun(@(x)x.DATA,curr_block_dir_loaded,'un',0); final_EEG = cat(2,final_EEG_dir{:});
final_EEG_idx_cell = mat2cell(1:length(curr_block_dir_loaded),1,ones(1,length(curr_block_dir_loaded)));
final_EEG_idx_dir = cellfun(@(x,y)repmat(y,[1 size(x.DATA,2)]),curr_block_dir_loaded,final_EEG_idx_cell,'un',0); final_EEG_idx = cat(2,final_EEG_idx_dir{:});

% Concatenate RDA BLOCKs into one vector:
final_EEG_BLOCKS_dir = cellfun(@(x)x.DATA_BLOCKS,curr_block_dir_loaded,'un',0); final_EEG_BLOCKS = cat(2,final_EEG_BLOCKS_dir{:});
final_EEG_BLOCKS_dir_unique = cellfun(@(x)unique(x),final_EEG_BLOCKS_dir,'un',0); final_EEG_BLOCKS_dir_unique_cell = cellfun(@(x)mat2cell(x,1,ones(1,length(x))),final_EEG_BLOCKS_dir_unique,'un',0);
final_EEG_BLOCKS_dir_find = cellfun(@(x,y)arrayfun(@(z)(y == z),x,'un',0),final_EEG_BLOCKS_dir_unique,final_EEG_BLOCKS_dir,'un',0);
% final_EEG_BLOCKS_dir_idx = cellfun(@(x,y)cellfun(@(m,n) double(m)*sum(n) + [1:sum(n)]-1,x,y,'un',0),final_EEG_BLOCKS_dir_unique_cell,final_EEG_BLOCKS_dir_find,'un',0);
final_EEG_BLOCKS_dir_idx = cellfun(@(x,y)cellfun(@(m,n) double(m-1)*sum(n) + [1:sum(n)],x,y,'un',0),final_EEG_BLOCKS_dir_unique_cell,final_EEG_BLOCKS_dir_find,'un',0);

final_EEG_BLOCKS_dir_mod = final_EEG_BLOCKS_dir;
for i = 1:length(final_EEG_BLOCKS_dir_mod)
    for j = 1:length(final_EEG_BLOCKS_dir_idx{i})
        curr_idx = final_EEG_BLOCKS_dir_find{i}{j};
        final_EEG_BLOCKS_dir_mod{i}(curr_idx) = final_EEG_BLOCKS_dir_idx{i}{j};
    end
end
final_EEG_BLOCKS_mod = cat(2,final_EEG_BLOCKS_dir_mod{:}); % These are the indices of the EEG data in terms of the RDA blocks

%                         final_EEG_BLOCKS_unique = unique(final_EEG_BLOCKS); final_EEG_BLOCKS_unique_bin = cell(1,length(final_EEG_BLOCKS_unique_bin));
%                         for i = 1:length(final_EEG_BLOCKS_unique)
%                             final_EEG_BLOCKS_unique_bin{i} = final_EEG_BLOCKS==final_EEG_BLOCKS_unique(i);
%                         end
%                         cellfun(final_EEG_BLOCKS_unique_bin)
% final_EEG_BLOCKS_idx =

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Doesn't work - creates some events out of bounds %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         % Concatenate Markers into one vector (with respect to the first index - final_EEG_BLOCKS_mod(1)):
%                         final_EEG_EVENT_dir = cellfun(@(x)x.EVENT,curr_block_dir_loaded,'un',0); final_EEG_EVENT = cat(2,final_EEG_EVENT_dir{:});
%                         RDA_blocksize = final_EEG_EVENT(1).RDAblocksize;
%                         % final_EEG_offset = mat2cell(cumsum(cellfun(@(x) length(x),final_EEG_idx_dir)) - length(final_EEG_idx_dir{1}),1,ones(1,length(curr_block_dir_loaded)));
%                         % final_EEG_EVENT_dir_sample_mod = cellfun(@(x,y) arrayfun(@(z) y + z.sample,x),final_EEG_EVENT_dir,final_EEG_offset,'un',0);
%
%                         for i = 1:length(final_EEG_EVENT)
%                             final_EEG_EVENT(i).latency = ((final_EEG_EVENT(i).RDAblock)*RDA_blocksize+final_EEG_EVENT(i).RDAposition) - final_EEG_BLOCKS_mod(1);
%                             final_EEG_EVENT(i).code = final_EEG_EVENT(i).type; final_EEG_EVENT(i).type = {final_EEG_EVENT(i).value};
%                         end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Concatenate Markers into one vector (with respect to the first index - final_EEG_BLOCKS_mod(1)):
final_EEG_EVENT_dir = cellfun(@(x)x.EVENT,curr_block_dir_loaded,'un',0); final_EEG_EVENT = cat(2,final_EEG_EVENT_dir{:});
RDA_blocksize = final_EEG_EVENT(1).RDAblocksize;

% Fix latency issues in the first volume (caused by the get_EEG_data_fMRIsync_RDA_Ver2 running for a while before the first slice marker comes in):
final_EEG_EVENT_firstLATENCY = (cell2mat({final_EEG_EVENT_dir{1}(:).RDAblock})-1)*RDA_blocksize + cell2mat({final_EEG_EVENT_dir{1}(:).RDAposition}); % Correct for latencies in the first block
final_EEG_EVENT_dir_mod = final_EEG_EVENT_dir;
for i = 1:length(final_EEG_EVENT_firstLATENCY)
    curr_find = find(final_EEG_BLOCKS_dir_mod{1} == final_EEG_EVENT_firstLATENCY(i)); % curr_find is the index with respect to the start of the first datapoint
    if ~isempty(curr_find) final_EEG_EVENT_dir_mod{1}(i).sample = uint32(curr_find); else final_EEG_EVENT_dir_mod{1}(i).sample = uint32(1); end
end

% Compute latency of all following events with respect to the first datapoint:
% final_EEG_offset = mat2cell(cumsum(cellfun(@(x) length(x),final_EEG_idx_dir)) - length(final_EEG_idx_dir{1}),1,ones(1,length(curr_block_dir_loaded)));
final_EEG_offset = circshift(cellfun(@(x) length(x),final_EEG_idx_dir),1); final_EEG_offset(1) = 0;
final_EEG_offset = mat2cell(cumsum(final_EEG_offset),1,ones(1,length(curr_block_dir_loaded)));
% final_EEG_EVENT_LATENCY_dir = cellfun(@(x,y)(cell2mat({x(:).sample})) + y - 1 ,final_EEG_EVENT_dir_mod,final_EEG_offset,'un',0);
final_EEG_EVENT_LATENCY_dir = cellfun(@(x,y)(cell2mat({x(:).sample})) + y ,final_EEG_EVENT_dir_mod,final_EEG_offset,'un',0);
final_EEG_EVENT_LATENCY = cat(2,final_EEG_EVENT_LATENCY_dir{:});

% final_EEG_EVENT_dir_sample_mod = cellfun(@(x,y) arrayfun(@(z) y + z.sample,x),final_EEG_EVENT_dir,final_EEG_offset,'un',0);

for i = 1:length(final_EEG_EVENT)
    % final_EEG_EVENT(i).latency = ((final_EEG_EVENT(i).RDAblock)*RDA_blocksize+final_EEG_EVENT(i).RDAposition) - final_EEG_BLOCKS_mod(1);
    final_EEG_EVENT(i).latency = final_EEG_EVENT_LATENCY(i);
    final_EEG_EVENT(i).code = final_EEG_EVENT(i).type; final_EEG_EVENT(i).type = {final_EEG_EVENT(i).value};
end

% Pad final_EEG data with extra values before the onset and after the end to allow for GA filtering:
pad_amount = 250;
final_EEG = padarray(final_EEG',pad_amount,'pre')';
final_EEG = padarray(final_EEG',pad_amount,'post')';
for i = 1:length(final_EEG_EVENT) final_EEG_EVENT(i).latency = final_EEG_EVENT(i).latency + pad_amount; end

% Assign output structure:
final_data = [];
final_data.final_EEG = final_EEG;
final_data.final_EEG_EVENT = final_EEG_EVENT;
final_data.final_EEG_BLOCKS_dir = final_EEG_BLOCKS_dir;
final_data.final_EEG_EVENT_dir_mod = final_EEG_EVENT_dir_mod;
final_data.final_EEG_EVENT_LATENCY_dir = final_EEG_EVENT_LATENCY_dir;
final_data.final_EEG_BLOCKS_dir_mod = final_EEG_BLOCKS_dir_mod;

%% This section moved to curate_mat2set_EEGfMRI - since the final_EEG datafile needs to be in the 'base' workspace:
% % Import into EEGLAB format:
% EEG = pop_importdata('dataformat','array','nbchan',0,'data','final_EEG','srate',srate,'pnts',0,'xmin',0);
% EEG = add_events_from_latency_EEGfMRI(EEG,{final_EEG_EVENT(:).type}, cell2mat({final_EEG_EVENT(:).latency}),cell2mat({final_EEG_EVENT(:).duration}));
% 
% EEG = pop_chanedit(EEG,'load',{chanlocs_file 'filetype' 'autodetect'});
% EEG.setname = [dataset_name '_MAT']; EEG = eeg_checkset( EEG );
% 
% % Detect discontinuities and insert "boundary" events:
% total_num_vols = length(final_EEG_BLOCKS_dir); pnts_per_vol = scan_parameters.TR*srate;
% data_breaks = find(cellfun(@(x) length(x),final_EEG_idx_dir) < pnts_per_vol)'; data_breaks_idx = zeros(length(data_breaks),2);
% for i = 1:length(data_breaks)
%     curr_i = find(arrayfun(@(x)strcmp(x,scan_parameters.slice_marker),{final_EEG_EVENT_dir_mod{data_breaks(i)}.value}),1,'first');
%     data_breaks_idx(i,:) = [final_EEG_EVENT_LATENCY_dir{data_breaks(i)}(curr_i) final_EEG_EVENT_LATENCY_dir{data_breaks(i)}(curr_i)];
% end
% [EEG.event] = eeg_insertbound(EEG.event, EEG.pnts, data_breaks_idx);
% EEG = eeg_checkset(EEG, 'eventconsistency');


