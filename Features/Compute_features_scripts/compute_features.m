% Compute features from the preprocessed dataset

% Set features to compute:
% feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'STE', 'FBCSP', 'BPow', 'CFC_SI'};
% feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};

% Set Cluster Properties:
par_window = 1;
if isempty(gcp('nocreate'))
    numCores = feature('numcores')
    p = parpool(numCores);
end

% Settings for Computing EEG features:
% Initialize flags to zeros:
compute_COH = 0; compute_PAC = 0; compute_PLI = 0; compute_dPLI = 0;
compute_STE = 0; compute_FBCSP = 0; compute_BPow = 0; compute_Bspec = 0;
compute_CFC_SI = 0; compute_PLV = 0; 

% Options for FBCSP
fbcsp_opt.features.cspcomps = 2;      % This is K from the lecture slides (you should have 2*K features per filter in the filter bank)
fbcsp_opt.features.cspband = [4,32];  % The filter bank will be constructed within this frequency range
fbcsp_opt.features.cspstep = 2;       % Adjacent filters will having initial frequency shifted by 2 Hz
fbcsp_opt.features.cspwidth = 4;      % Each bandpass filter will cover a 4 Hz band
fbcsp_opt.mode = 'concat';            % CSP can actually be computed according to about 4 mathematically equivalent formulations. I've selected one for you to use here.
fbcsp_opt.filter.type = 'butter';     % The filter bank will be constructed of Butterworth bandpass filters
fbcsp_opt.filter.order = 4;           % Each filter will be second-order
fbcsp_opt.filter.sr = EEG.srate;      % This is just the sampling rate of the EEG signal (needed to compute the coefficients of each bandpass filters)

% Options for Bispectrum
downsample_factor = 16;

% Options for Windowing:
windowLength = [100]; % Could be multiple windowLengths [100,200,400,1000] 
windowStep = [50]; % Could be multiple windowSteps [50, 100, 200, 500]
numOfProt = length(windowLength);
if length(size(EEG.data)) > 2 num_epochs = size(EEG.data,3);  else num_epochs = 1; end 

% Options for frequency bands to compute features over:
frequency_bands = {'full','delta','theta','alpha','beta','gamma','high_gamma'};

% Options for PLI + dPLI:
pli_opt.data_length_factor = 5;
pli_opt.permutation = 10; % Was originally 20
pli_opt.p_value = 0.05;

% Options for PAC:
% Should be in this order - {'full','delta','theta','alpha','beta','gamma','high_gamma'}
pac_opt.frequency_bands = frequency_bands;
pac_opt.numberOfBin = 10;
pac_opt.data_length_factor = 5;

% Options for CFC_SI:
% Should be in this order - {'full','delta','theta','alpha','beta','gamma','high_gamma'}
CFCSI_opt.frequency_bands = frequency_bands;

%% Decide which features to compute:
% Make output folder if not made:
if isempty(dir([curr_dir filesep 'EEG_Features']))
    mkdir([curr_dir filesep 'EEG_Features']);
end

% Identify which features are already computed:
if already_computed already_computed_feats = whos('-file',[curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_Epoch1.mat']); else  already_computed_feats = []; already_computed_feats.name = ''; end

for m = 1:length(feature_names)
    curr_feature = feature_names{m};
    switch curr_feature   
        
        case 'FBCSP' % Compute FBCSP features:            
            compute_FBCSP = isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_AllEpochs_' 'FBCSP']));
            
         case 'BPow' % Compute Frequency Band powers:
            compute_BPow = isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_AllEpochs_' 'BandPowers'])); 
        
        case 'COH'
            compute_COH = ~ismember('avg_coherence',{already_computed_feats(:).name});
            
        case 'PAC'
            compute_PAC = ~ismember('arr_sortamp',{already_computed_feats(:).name});
            
        case 'PLI'
            compute_PLI = ~ismember('PLI',{already_computed_feats(:).name});
            
        case 'dPLI'
            compute_dPLI = ~ismember('dPLI',{already_computed_feats(:).name});
            
        case 'STE'
            compute_STE = ~ismember('STE',{already_computed_feats(:).name});
            
        case 'Bspec' % Compute BiSpectrum:
            compute_Bspec = ~ismember('Bspec_features',{already_computed_feats(:).name});
            
        case 'CFC_SI' % Compute Cross-Frequency Coupling Synchronization Index:
            compute_CFC_SI = ~ismember('CFC_SI',{already_computed_feats(:).name});
            
        case 'PLV' % Compute Phase Locking Value:
            compute_PLV = 1;            
    end
end
runFeatureComputation = compute_COH + compute_PAC + compute_PLI + compute_dPLI + compute_STE + compute_FBCSP + compute_BPow + compute_Bspec + compute_CFC_SI + compute_PLV;

%% Prepare the dataset for feature computation:
if runFeatureComputation
    
    disp(['***************************** Computing Features *****************************']);
    
    % Prefilter the dataset into frequency bands +
    % Precompute the hilbert transform for use in PLI, dPLI, PAC, CFC_SI, PLV:
    EEGdata_filt = cell(1,length(frequency_bands));
    EEGdata_filt_hilbert = cell(1,length(frequency_bands));
    
    for i = 1:length(frequency_bands)
        freq_band = bandWidth(frequency_bands{i}); freq_low = freq_band(1); freq_high = freq_band(2);
        if length(size(EEG.data)) > 2
			EEGdata_filt{i} = bpfilter(freq_low, freq_high, EEG.srate, permute(double(EEG.data),[2, 1, 3])); % Dimension should be time x channels x epochs (so it will work on time)
		else
			EEGdata_filt{i} = bpfilter(freq_low, freq_high, EEG.srate, permute(double(EEG.data),[2, 1])); % Dimension should be time x channels (so it will work on time)
		end
        
        % Compute Hilbert transform:
		if length(size(EEG.data)) > 2
			curr_hilbert = zeros(size(EEGdata_filt{i}));
			parfor j = 1:size(EEGdata_filt{i},3)
				curr_hilbert(:,:,j) = hilbert(squeeze(EEGdata_filt{i}(:,:,j)));
			end
		else
			curr_hilbert = hilbert(EEGdata_filt{i});
		end
        EEGdata_filt_hilbert{i} = curr_hilbert;
    end
    
	if length(size(EEG.data)) > 2
		EEGdata_filt = cellfun(@(x) permute(x,[2, 1, 3]),EEGdata_filt,'UniformOutput',0);
		EEGdata_filt_hilbert = cellfun(@(x) permute(x,[2, 1, 3]),EEGdata_filt_hilbert,'UniformOutput',0);
    else
		EEGdata_filt = cellfun(@(x) permute(x,[2, 1]),EEGdata_filt,'UniformOutput',0);
		EEGdata_filt_hilbert = cellfun(@(x) permute(x,[2, 1]),EEGdata_filt_hilbert,'UniformOutput',0);	
	end
	
	
    %% Compute Features:
    % First compute FBCSP and Band Power across the whole dataset:
    if compute_FBCSP
        % Convert EEG data to cell for FBCSP computation:
        curr_EEG_cell = [];
        for i = 1:size(EEG.data,3) curr_EEG_cell{i} = squeeze(EEG.data(:,:,i)); end
        
        % [EEGcsp,CSP_features,cspftrs,Psel] = FBCSPfeatures(curr_EEG_cell,EEG.event,1:length(EEG.event),opt);
        [EEGcsp,CSP_features,cspftrs,Psel] = FBCSPfeatures(curr_EEG_cell,trial_data_num,1:length(trial_data_num),fbcsp_opt);
        % convert saveed variables to this! - analyzedData
        save([curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_AllEpochs_' 'FBCSP'],'EEGcsp','CSP_features','cspftrs','Psel');% Save the featureeeeeeesssss
    end
    
    if compute_BPow [bandPower_features] = EEG_band_powers(EEG.data, EEG.srate); save([curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_AllEpochs_' 'BandPowers'],'bandPower_features'); end
    
    % Compute features over the various sliding window protocols for the various Epochs
    disp(['***************************** Starting Feature Computation *****************************']);
    
    numFeatures = length(feature_names);
    % Features = cell(1,num_epochs);
    for j = 1:num_epochs
        
        disp(['Started processing Epoch ', num2str(j)]);
		if length(size(EEG.data)) > 2
			dataEpoch = cellfun(@(x)squeeze(x(:,:,j)),EEGdata_filt,'UniformOutput',0);
			dataEpoch_hilbert = cellfun(@(x)squeeze(x(:,:,j)),EEGdata_filt_hilbert,'UniformOutput',0);
		else
			dataEpoch = EEGdata_filt; dataEpoch_hilbert = EEGdata_filt_hilbert;		
		end
        
        % Initialize variables:
        if compute_COH all_coherence = cell(1,numOfProt); avg_coherence = cell(1,numOfProt); end
        if compute_PAC arr_sortamp = cell(1,numOfProt); arr_plotamp = cell(1,numOfProt); arr_avgamp = cell(1,numOfProt); end
        if compute_PLI PLI = cell(1,numOfProt); PLIcorr = cell(1,numOfProt); PLI_z_score = cell(1,numOfProt); end
        if compute_dPLI dPLI = cell(1,numOfProt); dPLIcorr = cell(1,numOfProt); dPLI_z_score = cell(1,numOfProt); end
        if compute_STE STE = cell(1,numOfProt); NSTE = cell(1,numOfProt); end
        if compute_Bspec Bspec_features_raw_cell = cell(1,numOfProt); Bspec_features_real_cell = cell(1,numOfProt); Bspec_features = cell(1,numOfProt); end
        if compute_CFC_SI CFC_SI = cell(1,numOfProt); CFC_SI_mag = cell(1,numOfProt); CFC_SI_theta = cell(1,numOfProt); end
        
        tic; analyzedData = [];
        for i = 1:numOfProt % Iterate over different windowLength protocols
            if compute_COH [avg_coherence{i}, all_coherence{i}] = coherence_custom(dataEpoch,windowLength(i),windowStep(i),EEG.srate,par_window); end
            if compute_PAC [arr_sortamp{i}, arr_plotamp{i}, arr_avgamp{i}] = phase_amp_couple_custom(dataEpoch_hilbert, windowLength(i),windowStep(i),pac_opt,par_window); end
            if compute_PLI [PLI{i}, PLIcorr{i}, PLI_z_score{i}] = pli_custom(dataEpoch_hilbert,windowLength(i),windowStep(i),pli_opt,par_window); end
            if compute_dPLI [dPLI{i}, dPLIcorr{i}, dPLI_z_score{i}] = dpli_custom(dataEpoch_hilbert,windowLength(i),windowStep(i),pli_opt,par_window); end
            if compute_STE [STE{i}, NSTE{i}] = symbolicTransferEntropy(dataEpoch,windowLength(i),windowStep(i)); end
            if compute_Bspec [Bspec_features_raw_cell{i}, Bspec_features_real_cell{i}, Bspec_features{i}] = bispectrum(dataEpoch,windowLength(i),windowStep(i)); end
            if compute_CFC_SI [CFC_SI{i}, CFC_SI_mag{i}, CFC_SI_theta{i}] = CFC_SynchronizationIndex(dataEpoch_hilbert, windowLength(i),windowStep(i),CFCSI_opt,par_window); end
            
        end
        
        if compute_COH analyzedData.all_coherence = all_coherence; analyzedData.avg_coherence = avg_coherence; end
        if compute_PAC analyzedData.arr_sortamp = arr_sortamp; analyzedData.arr_plotamp = arr_plotamp; analyzedData.arr_avgamp = arr_avgamp; end
        if compute_PLI analyzedData.PLI = PLI; analyzedData.PLIcorr = PLIcorr; analyzedData.PLI_z_score = PLI_z_score; end
        if compute_dPLI analyzedData.dPLI = dPLI; analyzedData.dPLIcorr = dPLIcorr; analyzedData.dPLI_z_score = dPLI_z_score; end
        if compute_STE analyzedData.STE = STE; analyzedData.NSTE = NSTE; end
        if compute_Bspec analyzedData.Bspec_features_raw_cell = Bspec_features_raw_cell; analyzedData.Bspec_features_real_cell = Bspec_features_real_cell; analyzedData.Bspec_features = Bspec_features; end
        if compute_CFC_SI analyzedData.CFC_SI = CFC_SI; analyzedData.CFC_SI_mag = CFC_SI_mag; analyzedData.CFC_SI_theta = CFC_SI_theta; end
        
        disp(['Finished processing Epoch ', num2str(j), ' in ', num2str(toc), 's.']);
        
        % Features{j} = analyzedData;
        
        filename = [curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_Epoch' num2str(j)];
        if already_computed parsave_struct(filename, analyzedData, 1); else parsave_struct(filename, analyzedData, 0); end
    end
    
else
    disp(['***************************** Features already computed *****************************']);
end

%% Save all epochs accumulated in Features cell:
% final_filename = [curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(subIdx) '_Ses' num2str(sesIdx) '_AllEpochs'];
% save(final_filename, 'Features', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% %% Compute Bispectrum features:
% tic
% 
% parfor i = 1:size(EEG.data,3)
%     [Bspec_curr, waxis] = bispecd(squeeze(EEG.data(:,:,i))');
%     Bspec_features_raw_cell{i} = avgpool(Bspec_curr,downsample_factor);
%     Bspec_features_real_cell{i} = reshape(real(Bspec_features_raw_cell{i}),[size(Bspec_features_raw_cell{i},1)*size(Bspec_features_raw_cell{i},2), 1]);
%     % Bspec_features_imag_cell{i} = reshape(imag(Bspec_features_raw_cell{i}),[1,size(Bspec_features_raw_cell{i},1)*size(Bspec_features_raw_cell{i},2)]);
%     % Bspec_features_cell{i} = [Bspec_features_real_cell{i}, Bspec_features_imag_cell{i}];
%     Bspec_features_cell{i} = [Bspec_features_real_cell{i}];
% end
% Bspec_features = cell2mat(Bspec_features_cell);
% toc
% % Bspec_features;    
% %%
% save([curr_dir filesep 'Features'],'EEGcsp','cspftrs','Psel','CSP_features',...
%     'bandPower_features',...
%     'Bspec_features', 'Bspec_features_raw_cell', 'waxis')

% Concatenate all the features into one vector
    % Features = cat(1,CSP_features,bandPower_features)';
    
    % If you wish to use feature selection, that should go here
    
    % Classification - put your classifier here (get Yhat from your
    % classifier). If your classifier instead just outputs a classification
    % accuracy, you can remove the variables 'Yhat' and 'Correct' and fill
    % the elements of Accuracy directly from your classifier function
    % [TrainAccuracy, TestAccuracy, Model] = classify_SVM_libsvm(Features,trial_data_num','RBF',0.1,trainidx,testidx);
%     MdlSVM_curr = fitcsvm(Features(trainidx,:),trial_data_num(trainidx)','Standardize',true,...
%         'KernelFunction','RBF','KernelScale','auto');
%     MdlSVM = fitPosterior(MdlSVM_curr);
%     
%     % Test the SVM Model:
%     [YTrainhat] = resubPredict(MdlSVM);
%     % [YTrainhat,YTrainhat_NegLoss,YTrainhat_score,YTrainhat_posterior] = resubPredict(MdlSVM);
%     [YTesthat,YTesthat_score] = predict(MdlSVM,Features(testidx,:));

%% Old Arachaic code:
% if compute_PLI [PLI{i}, PLIcorr{i}, PLI_z_score{i}] = pli(squeeze(EEG.data(:,:,j)),windowLength(i),windowStep(i)); end