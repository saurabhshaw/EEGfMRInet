% GENERATE_GA_TRIGS Generates the triggers for marking the gradient
% periodicity for GA filtering purposes
%
% Usage
%    Trigs = GENERATE_GA_TRIGS(EEG_data, cfg)
%
% Input
%    EEG_data (M x N matrix): The segment of EEG to be used for trigger
%                             generation
%                                   M - Number of Channels
%                                   N - Number of Samples
%
%    cfg (struct)           : Structure containing the configuration
%                             parameters for the script
%                                   cfg.Fs         - Sampling rate
%                                   cfg.plot_img   - Flag to display/hide plots
%                                   cfg.TR         - Repetition time (sec)
%                                   cfg.num_Slices - Slices per TR
%                                   cfg.profile    - Flag to display/hide profiling data 
%
% Output
%    Trigs (1xK matrix): The vector containing the triggers for GA
%                        filtering (K triggers).
% 
% See also
%    POP_FMRIB_FASTR, FMRIB_FASTR
%
% Author: Saurabh B Shaw
%

function Trigs = generate_GA_trigs(EEG_data, cfg)

% Set defaults:
if ~isfield(cfg, 'Fs'),               cfg.Fs = 5000;         end
if ~isfield(cfg, 'plot_img'),         cfg.plot_img = 1;      end
if ~isfield(cfg, 'TR'),               cfg.TR = 2;            end
if ~isfield(cfg, 'num_Slices'),       cfg.num_Slices = 39;   end
if ~isfield(cfg, 'profile'),          cfg.profile = 0;       end
if ~isfield(cfg, 'peak_classify'),    cfg.peak_classify = 1; end

Fs = cfg.Fs; plot_img = cfg.plot_img; TR = cfg.TR; 
num_Slices = cfg.num_Slices; do_profile = cfg.profile;
search_peak_types = cfg.peak_classify;

if do_profile tic; end

%% Custom Algorithm:
% Zero-mean and Normalize the signal:
EEG_data = EEG_data - repmat(mean(EEG_data,2),[1,size(EEG_data,2)]);
EEG_data = EEG_data./repmat(max(EEG_data,[],2),[1,size(EEG_data,2)]);
time = (1:size(EEG_data,2))./Fs;

% Detect the peaks:
X = double(EEG_data(1,:));
[pklg,lclg] = findpeaks(X, ...
    'MinPeakDistance',floor(TR/num_Slices)*Fs,'MinPeakheight',0.6);

if abs(max(pklg) - min(pklg)) > 0.13 % For situations when the smaller peaks are not detected at all
    
    %% Identify the different kinds of peaks present:
    if search_peak_types
        max_peak_types = 3;
        idx = cell(1,max_peak_types); silh = cell(1,max_peak_types);
        h = cell(1,max_peak_types); mean_silh = cell(1,max_peak_types);
        loc = cell(1,max_peak_types); D = cell(1,max_peak_types);
        m_D = cell(1,max_peak_types);
        parfor i = 1:max_peak_types
            [idx{i}, loc{i},m_D{i},D{i}] = kmeans(pklg',i,'distance','cityblock');
            [silh{i},h{i}] = silhouette(pklg',idx{i},'cityblock');
            mean_silh{i} = mean(silh{i});
        end
        close all;
        [~,num_peak_types] = max(cell2mat(mean_silh));
        
    else
        num_peak_types = 2; i = num_peak_types;
        [idx{i}, loc{i},m_D{i},D{i}] = kmeans(pklg',num_peak_types,'distance','cityblock');
        [silh{i},h{i}] = silhouette(pklg',idx{i},'cityblock');
        close all;
    end
    
    %% Find the indices of the highest peak identified:
    % Find the highest peak of the peaks identified:
    lclg_peaks = cell(1,num_peak_types); pklg_peaks = cell(1,num_peak_types);
    mean_pklg = zeros(1,num_peak_types);
    parfor j = 1:num_peak_types
        curr_peak_vect = (idx{num_peak_types} == j);
        lclg_peaks{j} = lclg(curr_peak_vect);
        pklg_peaks{j} = pklg(curr_peak_vect);
        mean_pklg(j) = mean(pklg(curr_peak_vect));
    end
    [~,max_pklg_idx] = max(mean_pklg);
    
    if do_profile toc; end
    
    %% Plot the image:
    if plot_img
        figure; plot(time,X);
        hold on;
        pks = plot(time(lclg_peaks{max_pklg_idx}),pklg_peaks{max_pklg_idx}+0.05,'vk');
    end
    
    Trigs = lclg_peaks{max_pklg_idx};
else
    Trigs = lclg;
end

% %% Compute Autocorrelation such that it is unity at zero lag:
% [autocor,lags] = xcorr(double(EEG_data(1:2*Fs)),'coeff');
% 
% if plot_img
%     figure; plot(lags/Fs,autocor); xlabel('Lag (Seconds)'); ylabel('Autocorrelation');
% end
% 
% %% Find the peak distances:
% [pksh,lcsh] = findpeaks(autocor);
% short = mean(diff(lcsh))/Fs;
% 
% [pklg,lclg] = findpeaks(autocor, ...
%     'MinPeakDistance',ceil(TR/num_Slices)*Fs,'MinPeakheight',0.3);
% long = mean(diff(lclg))/Fs;
% 
% if plot_img
%     hold on
%     pks = plot(lags(lcsh)/Fs,pksh,'or', ...
%         lags(lclg)/Fs,pklg+0.05,'vk');
%     hold off
%     legend(pks,[repmat('Period: ',[2 1]) num2str([short;long],0)]);    
% end