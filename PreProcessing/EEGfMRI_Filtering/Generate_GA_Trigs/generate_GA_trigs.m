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
if ~isfield(cfg, 'fudge_fac'),        cfg.fudge_fac = 15;    end


Fs = cfg.Fs; plot_img = cfg.plot_img; TR = cfg.TR; 
num_Slices = cfg.num_Slices; do_profile = cfg.profile;
search_peak_types = cfg.peak_classify; fudge_fac = cfg.fudge_fac;

if do_profile tic; end

%% Custom Algorithm:
% Zero-mean and Normalize the signal:
EEG_data = EEG_data - repmat(mean(EEG_data,2),[1,size(EEG_data,2)]);
EEG_data = EEG_data./repmat(max(EEG_data,[],2),[1,size(EEG_data,2)]);
time = (1:size(EEG_data,2))./Fs;

% Detect the peaks:
X = double(EEG_data(1,:));
[pklg,lclg] = findpeaks(X, ...
    'MinPeakDistance',floor(TR/num_Slices)*Fs,'MinPeakheight',0.6); % was 0.6

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

%% Double check for the correct peaks detected:
int_peak_samp = round((TR*Fs)/num_Slices); % The estimated inter-peak times
low_limit = int_peak_samp - fudge_fac; high_limit = int_peak_samp + fudge_fac;
Trigs_diff_mean = zeros(1,fudge_fac); Trigs_diff_std = zeros(1,fudge_fac);

% Check if peak duration is incorrect:
dur_test = circshift(Trigs',[-1,0])' - Trigs;
dur_test_mean = mean(dur_test(1:end-1));

if ((dur_test_mean > high_limit) || (low_limit > dur_test_mean))
    parfor i = 1:fudge_fac
        Trigs_shift = circshift(Trigs',[-i,0])';
        A = Trigs_shift - Trigs;
        Trigs_diff{i} = A;
        Trigs_diff_mean(i) = mean(A(1:end-i));
        Trigs_diff_std(i) = std(A(1:end-i));
    end
    corr_peak_shift = find((Trigs_diff_mean < high_limit) & (low_limit <= Trigs_diff_mean));
    corr_vect = 1:corr_peak_shift:length(Trigs);
    corr_Trigs = Trigs(corr_vect);
    
%     % Check for inconsistencies within the identified global shift:
%     corr_diff = Trigs_diff{corr_peak_shift}(1:end-corr_peak_shift); corr_diff = corr_diff./max(corr_diff);
%     [corr_p, corr_l] = findpeaks(corr_diff,'MinPeakheight',0.8);
%     
%     %% Plot the image:
%     if plot_img
%         figure; plot(time,X);
%         hold on;
%         pks = plot(time(lclg_peaks{max_pklg_idx}(corr_l)),pklg_peaks{max_pklg_idx}(corr_l)+0.05,'vk');
%     end
    
    Trigs = corr_Trigs;
        
    %% Plot the image:
    if plot_img
        figure; plot(time,X);
        hold on;
        pks = plot(time(lclg_peaks{max_pklg_idx}(corr_vect)),pklg_peaks{max_pklg_idx}(corr_vect)+0.05,'vk');
    end
    
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