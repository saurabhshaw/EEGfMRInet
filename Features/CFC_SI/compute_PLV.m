function [SI] = compute_PLV(subject_number,session_number,higher_freq_low,higher_freq_high,lower_freq_low,lower_freq_high)

%subject_number = 1;
%session_number = 1;


finalData = [];
windowLength = 0.5; % Time is seconds
windowOverlap = 0.5; % Overlap in percentage of windowLength

%alpha
% higher_freq_low = 73.5;
% higher_freq_high = 78.5;

%gamma
% lower_freq_low = 6;
% lower_freq_high = 12;

srate = 500; % 500 points per second

% Create the gamma band filter:
[higherFilt.b,higherFilt.a] = butter(2,[higher_freq_low/(srate/2) higher_freq_high/(srate/2)]);
[lowerFilt.b,lowerFilt.a] = butter(2,[lower_freq_low/(srate/2) lower_freq_high/(srate/2)]);

%% Loads participant data.

fileName = ['EEGfMRI_data_Subject',num2str(subject_number),'_Session',num2str(session_number),'.mat'];

load(fileName,'EEG_data_window','fMRI_labels_selected_window_thresh');

% get some params from EEG dataset
nchans = size(EEG_data_window,3);
N = size(EEG_data_window,2);
nepochs = size(EEG_data_window,1);

% Setup Windowing:
windowLength_num = windowLength*srate; % Mulitply by 2 because we take extra points on either side of the actual window, filter it, and then discard those points to avoid ringing.
windowOverlap_num = floor(windowOverlap*windowLength_num) ;
window_start_vect = 1:windowLength_num-windowOverlap_num:N;
window_end_vect = window_start_vect + windowLength_num - 1;
window_start_vect = window_start_vect(window_end_vect <= N);
window_end_vect = window_end_vect(window_end_vect <= N);
nfft = nextpow2(windowLength_num);
nwindows = length(window_end_vect);

% Plot
% will use for loop to go through each of the windows
%plot(plot_data(:,4));

hilbert_higherFiltdata = zeros(nepochs,nchans,nwindows,nfft);
power_higherFiltdata = zeros(nepochs,nchans,nwindows,nfft);
hilbertpower_higherFiltdata = zeros(nepochs,nchans,nwindows,nfft);
hilbert_lowerFiltdata = zeros(nepochs,nchans,nwindows,nfft);

%window_higherFiltdata = zeros(nchans,nwindows);
%window_lowerFiltdata = zeros(nchans,nwindows);

theta_higherFiltdata = zeros(nepochs,nchans,nwindows,nfft);
theta_lowerFiltdata = zeros(nepochs,nchans,nwindows,nfft);

SI = zeros(nepochs,nchans,nwindows);

for i = 1:nepochs
    
    tic
    %% Makes referring to the data and sampling rate easier
    data = squeeze(EEG_data_window(i,:,:));    
    
    % Get continuous data:
    A = data';
    
    % %    [test_window_pxx,fxx_window_data] = pwelch(A(1,window_start_vect(1):window_end_vect(1)),[],[],[],srate,'power');
    % [test_window_pxx,fxx_window_data] = pwelch(A(1,window_start_vect(1):window_end_vect(1)),floor(windowLength_num*0.25),floor(windowLength_num*0.125),nfft,srate,'power');
    %  %   [test_pxx,fxx_data] = pwelch(A(1,:),[],[],[],srate,'power');
    %
    %  %   alpha_window_idx = fxx_window_data > higher_freq_low & fxx_window_data < higher_freq_high;
    %  %   alpha_idx = fxx_data > higher_freq_low & fxx_data < higher_freq_high;
    %
    
    % higherFilt_window_idx = fxx_window_data > higher_freq_low & fxx_window_data < higher_freq_high;
    % lowerFilt_window_idx = fxx_window_data > lower_freq_low & fxx_window_data < lower_freq_high;
    
    % window_pxx_higherFiltdata = zeros(nchans,nwindows,length(test_window_pxx));
    % window_pxx_lowerFiltdata = zeros(nchans,nwindows,length(test_window_pxx));
    
    
    
    %     window_alpha = zeros(nchans,nwindows);
    %     window_FAA = zeros(1,nwindows);
    %
    %     pxx_data = zeros(nchans,length(test_pxx));
    %     alpha = zeros(1,nchans);
    
    %% loop thorugh channels and compute spectral power- update eegspec
    for j = 1:nwindows
        parfor chan = 1:nchans
            higherFilt_window_data = filtfilt(higherFilt.b,higherFilt.a,A(chan,window_start_vect(j):window_end_vect(j)));
            lowerFilt_window_data = filtfilt(lowerFilt.b,lowerFilt.a,A(chan,window_start_vect(j):window_end_vect(j)));
            
            % Calculate Hilbert transform (Real and Imaginary components)
            hilbert_higherFiltdata(i,chan,j,:) = hilbert(higherFilt_window_data,nfft);
            power_higherFiltdata(i,chan,j,:) = abs(hilbert_higherFiltdata(i,chan,j,:));
            hilbertpower_higherFiltdata(i,chan,j,:) = hilbert(squeeze(power_higherFiltdata(i,chan,j,:)),nfft);
            hilbert_lowerFiltdata(i,chan,j,:) = hilbert(lowerFilt_window_data,nfft);
            
            % Normalize and demean Hilbert transform
            %window_hilbert_higherFiltdata(chan,j,:) = window_hilbert_higherFiltdata(chan,j,:)./norm(squeeze(window_hilbert_higherFiltdata(chan,j,:))); window_hilbert_higherFiltdata(chan,j,:) = window_hilbert_higherFiltdata(chan,j,:) - mean(window_hilbert_higherFiltdata(chan,j,:));
            %window_hilbert_lowerFiltdata(chan,j,:) = window_hilbert_lowerFiltdata(chan,j,:)./norm(squeeze(window_hilbert_lowerFiltdata(chan,j,:))); window_hilbert_lowerFiltdata(chan,j,:) = window_hilbert_lowerFiltdata(chan,j,:) - mean(window_hilbert_lowerFiltdata(chan,j,:));
            
            % Calculate Phase from Real and Imaginary components:
            theta_higherFiltdata(i,chan,j,:) = atan(imag(hilbertpower_higherFiltdata(i,chan,j,:))./real(hilbertpower_higherFiltdata(i,chan,j,:)));
            theta_lowerFiltdata(i,chan,j,:) = atan(imag(hilbert_lowerFiltdata(i,chan,j,:))./real(hilbert_lowerFiltdata(i,chan,j,:)));
            
            % Calculate synchronization index (SI):
            SI(i,chan,j) = (1/windowLength_num).*sum(exp(1i.*(squeeze(theta_lowerFiltdata(i,chan,j,:)) - squeeze(theta_higherFiltdata(i,chan,j,:)))));
            
            %[window_pxx_higherFiltdata(chan,j,:),~] = pwelch(higherFilt_window_data,floor(windowLength_num*0.25),floor(windowLength_num*0.125),nfft,srate,'power');
            %[window_pxx_lowerFiltdata(chan,j,:),~] = pwelch(lowerFilt_window_data,floor(windowLength_num*0.25),floor(windowLength_num*0.125),nfft,srate,'power');
            
            %window_higherFiltdata(chan,j) = sum(window_pxx_higherFiltdata(chan,j,higherFilt_window_idx));
            %window_lowerFiltdata(chan,j) = sum(window_pxx_lowerFiltdata(chan,j,lowerFilt_window_idx));
        end
        
        % left_alpha = log10(mean(window_alpha(left_chans,j)));
        % right_alpha = log10(mean(window_alpha(right_chans,j)));
        % window_FAA(j) = (right_alpha-left_alpha)/(right_alpha+left_alpha);
    end
    toc
end

%     % For the whole segment of EEG:
%     for chan = 1:nchans
%         curr_data = filter(higherFilt.b,higherFilt.a,A(chan,:));        
%         [pxx_data(chan,:),fxx_data] = pwelch(curr_data,[],[],[],srate,'power');
%         alpha(chan) = mean(pxx_data(chan,alpha_window_idx));
%     end
%     L_alpha = log10(mean(alpha(left_chans)));
%     R_alpha = log10(mean(alpha(right_chans)));
%     FAA = (R_alpha-L_alpha)/(R_alpha+L_alpha);
    
% end

%% Plot Window Image:
%chan_num = 2; window_num = 2;
% % figure; plot(fxx_window_data,squeeze(window_pxx_higherFiltdata(chan_num,window_num,:)));
% % figure; plot(squeeze(window_lowerFiltdata(chan_num,:)));
%figure; plot(squeeze(abs(SI(chan_num,:))));