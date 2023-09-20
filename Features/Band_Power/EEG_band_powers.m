function [band_features] = EEG_band_powers(data, srate)
% This function calculates the band powers over all epochs 
% Call this function as:
%    [band_features] = EEG_band_powers(data, srate);
%
% Author: Saurabh Bhaskar Shaw
% 2019
%

%%
% clear all
freq_bands = {'alpha' 'beta' 'delta' 'theta' 'gamma'};
% freq_bands = {'alpha' 'beta'};

% Get some params from EEG dataset
nchans = size(data,1);
N = size(data,2);
nepochs = size(data,3);

% Setup Windowing:
windowLength_num = N;
nfft = 2^nextpow2(windowLength_num);
pwelch_window = floor(windowLength_num*0.2);
pwelch_overlap = floor(windowLength_num*0.1);

[test_pxx,fxx_data] = pwelch(data(1,:,1),pwelch_window,pwelch_overlap,nfft,srate,'power');

band_features = [];

%% Compute features for frequency bands:
for i = 1:length(freq_bands)
    [range] = bandWidth(freq_bands{i});
    band_low = range(1); band_high = range(2);
    
    % Create the band filter:
    [filt.b,filt.a] = butter(2,[band_low/(srate/2) band_high/(srate/2)]);
    
    alpha_idx = fxx_data > band_low & fxx_data < band_high;

    pxx_data = zeros(nchans,length(test_pxx),nepochs);
    band_power = zeros(nchans,nepochs);
    
    %% Compute band power for all epochs:
    parfor j = 1:nepochs
        curr_epoch_data = filtfilt(filt.b,filt.a,double(data(:,:,j)'));
        [curr_epoch_pxx,~] = pwelch(curr_epoch_data,pwelch_window,pwelch_overlap,nfft,srate,'power');
        curr_epoch_power = sum(curr_epoch_pxx(alpha_idx,:),1);
        pxx_data(:,:,j) = curr_epoch_pxx'; band_power(:,j) = curr_epoch_power';        
    end 
    
    band_features = cat(1,band_features,band_power);
end














function range = bandWidth(selection)

switch selection
    case 'full'
        range = [1 50];
    case 'alpha'
        range = [8 13];
    case 'beta'
        range = [13 30];
    case 'delta'
        range = [1 4];
    case 'theta'
        range = [4 8];
    case 'gamma'
        range = [30 50];
    otherwise
        range = [1 50];

end