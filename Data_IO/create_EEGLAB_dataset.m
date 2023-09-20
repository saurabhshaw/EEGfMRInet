% Create an EEGLAB dataset given a data vector
%
% This function is provided as is. It only works for some specific type of
% data. This is a simple function to help the developer and by no mean
% an all purpose function.

function [EEG] = create_EEGLAB_dataset(data, Fs, chanlocs, dataset_name)

load('empty_EEG_struct_EEGLAB.mat');
EEG.chanlocs = chanlocs;

% max_size = 1;
% for i = 1:size(data.trial,2)
%     trial_size = size(data.trial{i},2);
%     if trial_size > max_size
%         max_size = trial_size;
%     end
% end
% 
% EEG.data = single(NaN(size(data.trial{1},1),max_size,size(data.trial,2)));
% for i = 1:size(data.trial,2)
%   EEG.data(:,1:size(data.trial{i},2),i) = single(data.trial{i});
% end

EEG.data = data;
time_vect = (1:size(data,2))./Fs;

EEG.setname    = dataset_name;
EEG.filename   = '';
EEG.filepath   = '';
EEG.subject    = '';
EEG.group      = '';
EEG.condition  = '';
EEG.session    = [];
EEG.comments   = 'Raw_data';
EEG.nbchan     = size(data,1);
EEG.trials     = [];
EEG.pnts       = size(data,2);
EEG.srate      = Fs;
EEG.xmin       = time_vect(1);
EEG.xmax       = time_vect(end);
EEG.times      = time_vect;
EEG.ref        = []; %'common';
EEG.event      = [];
EEG.epoch      = [];
EEG.icawinv    = [];
EEG.icasphere  = [];
EEG.icaweights = [];
EEG.icaact     = [];
EEG.saved      = 'no';

[ALLEEG EEG CURRENTSET] = eeg_store([], EEG);

 