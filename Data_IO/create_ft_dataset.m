% CREATE_FT_DATASET Creates a Fieldtrip raw dataset
%
% Usage
%    ft_EEG = CREATE_FT_DATASET(raw_EEG, cfg)
%
% Input
%    raw_EEG (M x N matrix): The EEG data to be stored in the dataset. 
%                            M - Number of Channels
%                            N - Number of Samples
%
%    cfg (struct)          : The configuration structure that has the
%                            following fields -
%                               cfg.label - Labels for each Electrode
%                               cfg.Fs - The Sampling Frequency
%
% Output
%    ft_EEG (M x N matrix): The output fieldtrip EEG raw datastructure.
% 
% See also
%    EEGLAB2FIELDTRIP, EEGLAB2FIELDTRIP_EVENT
%
% Author: Saurabh B Shaw
%

function ft_EEG = create_ft_dataset(raw_EEG, cfg)

ft_EEG = []; ft_EEG.cfg = [];
ft_EEG.label = cfg.label; ft_EEG.fsample = cfg.Fs; ft_EEG.cfg.dataset = cfg.label;
ft_EEG.trial = num2cell(raw_EEG,[1,2]); ft_EEG.time = num2cell(([1:size(raw_EEG,2)]/cfg.Fs),[1,2]);
