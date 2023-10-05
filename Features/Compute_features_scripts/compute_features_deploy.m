% Compute features from the preprocessed dataset
function compute_features_deploy(input_mat)

% Set features to compute:
% feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'STE', 'FBCSP', 'BPow', 'CFC_SI'};
% feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};

% % Settings for Computing EEG features:
% 
% % Options for FBCSP
% fbcsp_opt.features.cspcomps = 2;      % This is K from the lecture slides (you should have 2*K features per filter in the filter bank)
% fbcsp_opt.features.cspband = [4,32];  % The filter bank will be constructed within this frequency range
% fbcsp_opt.features.cspstep = 2;       % Adjacent filters will having initial frequency shifted by 2 Hz
% fbcsp_opt.features.cspwidth = 4;      % Each bandpass filter will cover a 4 Hz band
% fbcsp_opt.mode = 'concat';            % CSP can actually be computed according to about 4 mathematically equivalent formulations. I've selected one for you to use here.
% fbcsp_opt.filter.type = 'butter';     % The filter bank will be constructed of Butterworth bandpass filters
% fbcsp_opt.filter.order = 4;           % Each filter will be second-order
% fbcsp_opt.filter.sr = EEG.srate;      % This is just the sampling rate of the EEG signal (needed to compute the coefficients of each bandpass filters)
% 
% % Options for Bispectrum
% downsample_factor = 16;
% 
% % Options for Windowing:
% windowLength = [100]; % Could be multiple windowLengths [100,200,400,1000] 
% windowStep = [50]; % Could be multiple windowSteps [50, 100, 200, 500]
% numOfProt = length(windowLength);
% if length(size(EEG.data)) > 2 num_epochs = size(EEG.data,3);  else num_epochs = 1; end 
% 
% % Options for frequency bands to compute features over:
% frequency_bands = {'full','delta','theta','alpha','beta','gamma','high_gamma'};
% 
% % Options for PLI + dPLI:
% pli_opt.data_length_factor = 5;
% pli_opt.permutation = 10; % Was originally 20
% pli_opt.p_value = 0.05;
% 
% % Options for PAC:
% % Should be in this order - {'full','delta','theta','alpha','beta','gamma','high_gamma'}
% pac_opt.frequency_bands = frequency_bands;
% pac_opt.numberOfBin = 10;
% pac_opt.data_length_factor = 5;
% pac_opt.bichannel = 1;
% 
% % Options for CFC_SI:
% % Should be in this order - {'full','delta','theta','alpha','beta','gamma','high_gamma'}
% CFCSI_opt.frequency_bands = frequency_bands;
% precompute_parfor = 0;
% save_all = 0;

%% Read in input data, Settings files and execute settings:
% Read in data:
load(input_mat);

% Read settings from file
fileID = fopen([base_path filesep 'Features' filesep 'Compute_features_scripts' filesep 'compute_features_deploy_settings.txt'],'r');
data = textscan(fileID,'%s','Delimiter','\n');
fclose(fileID);

% Execute statements:
for line_num = 1:length(data{1})
    eval(data{1}{line_num});    
end

%% Set Cluster Properties:
par_window = 1;
if isempty(gcp('nocreate'))
    numCores = feature('numcores')
    p = parpool(numCores);
end

%% Decide which features to compute:
% Initialize flags to zeros:
compute_COH = zeros(1,num_epochs); compute_PAC = zeros(1,num_epochs); compute_PLI = zeros(1,num_epochs); compute_dPLI = zeros(1,num_epochs);
compute_STE = zeros(1,num_epochs); compute_FBCSP = 0; compute_BPow = 0; compute_Bspec = zeros(1,num_epochs);
compute_CFC_SI = zeros(1,num_epochs); compute_PLV = zeros(1,num_epochs); 

% Make output folder if not made:
if isempty(dir([curr_dir filesep 'EEG_Features']))
    mkdir([curr_dir filesep 'EEG_Features']);
end

% Identify which epochs and features are already computed:
currFeatures_dir = dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_Epoch*.mat']);
% currEpochs_finished = cellfun(@(x) strsplit(x,{'Epoch','.mat'}),{currFeatures_dir.name},'un',0); currEpochs_finished = cellfun(@(x) str2num(x{2}),currEpochs_finished);

currEpochs_finished = cellfun(@(x) strsplit(x,{'Epoch'}),{currFeatures_dir.name},'un',0); 
currEpochs_finished = cellfun(@(x) strsplit(x{2},{'.mat'}),currEpochs_finished,'un',0);
currEpochs_finished = cellfun(@(x) str2num(x{1}),currEpochs_finished);

for i = 1:num_epochs already_computed_feats{i} = []; already_computed_feats{i}.name = ''; end
for i = 1:length(currFeatures_dir) 
    try 
        already_computed_feats{currEpochs_finished(i)} = whos('-file',[currFeatures_dir(i).folder filesep currFeatures_dir(i).name]);
        currFeatures_dir_OK(i) = true;
    catch
        currFeatures_dir_OK(i) = false;
    end
end
if ~isempty(currEpochs_finished) epochs_to_process = setdiff(1:num_epochs,currEpochs_finished(currFeatures_dir_OK)); end
% already_computed = ~isempty(epochs_to_process);
already_computed = cellfun(@(x)any(arrayfun(@(y)~isempty(y.name),x)),already_computed_feats);

for m = 1:length(feature_names)
    curr_feature = feature_names{m};
    switch curr_feature   
        
        case 'FBCSP' % Compute FBCSP features:            
            compute_FBCSP = isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_AllEpochs_' 'FBCSP']));
            
        case 'BPow' % Compute Frequency Band powers:
            compute_BPow = isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_AllEpochs_' 'BandPowers'])); 
        
        case 'COH'
            compute_COH = cellfun(@(x)~ismember('avg_coherence',{x(:).name}),already_computed_feats);
            
        case 'PAC'
            compute_PAC = cellfun(@(x)~ismember('arr_avgamp',{x(:).name}),already_computed_feats);
            
        case 'PLI'
            compute_PLI = cellfun(@(x)~ismember('PLI_z_score',{x(:).name}),already_computed_feats);
            
        case 'dPLI'
            compute_dPLI = cellfun(@(x)~ismember('dPLI_z_score',{x(:).name}),already_computed_feats);
        
        case 'STE'
            compute_STE = cellfun(@(x)~ismember('STE',{x(:).name}),already_computed_feats);
            
        case 'Bspec' % Compute BiSpectrum:
            compute_Bspec = cellfun(@(x)~ismember('Bspec_features',{x(:).name}),already_computed_feats);
            
        case 'CFC_SI' % Compute Cross-Frequency Coupling Synchronization Index:
            compute_CFC_SI = cellfun(@(x)~ismember('CFC_SI',{x(:).name}),already_computed_feats);
            
        case 'PLV' % Compute Phase Locking Value:
            compute_PLV = 1;            
    end
end
runFeatureComputation = compute_COH + compute_PAC + compute_PLI + compute_dPLI + compute_STE + compute_FBCSP + compute_BPow + compute_Bspec + compute_CFC_SI + compute_PLV;

%% Prepare the dataset for feature computation:
if any(runFeatureComputation)
    
    disp(['***************************** Computing Features *****************************']);
    
    % Prefilter the dataset into frequency bands +
    % Precompute the hilbert transform for use in PLI, dPLI, PAC, CFC_SI, PLV:
    if (compute_COH + compute_PAC + compute_PLI + compute_dPLI + compute_STE + compute_Bspec + compute_CFC_SI + compute_PLV) > 0
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
                if precompute_parfor
                    parfor j = 1:size(EEGdata_filt{i},3)
                        curr_hilbert(:,:,j) = hilbert(squeeze(EEGdata_filt{i}(:,:,j)));
                    end
                else
                    for j = 1:size(EEGdata_filt{i},3)
                        curr_hilbert(:,:,j) = hilbert(squeeze(EEGdata_filt{i}(:,:,j)));
                    end
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
        % save data: prefiltered + precomputed hilbert transform
        save([curr_dir filesep 'EEG_Features' filesep 'data_prefiltered_precomputedHT.mat'],"EEGdata_filt_hilbert","EEGdata_filt");
    else
        load([curr_dir filesep 'EEG_Features' filesep 'data_prefiltered_precomputedHT.mat'],"EEGdata_filt_hilbert","EEGdata_filt");
    end
	
    %% Compute Features:
    disp(['***************************** Starting Feature Computation *****************************']);

    % First compute FBCSP and Band Power across the whole dataset:
    if compute_FBCSP
        % Convert EEG data to cell for FBCSP computation:
        curr_EEG_cell = [];
        for i = 1:size(EEG.data,3) curr_EEG_cell{i} = squeeze(EEG.data(:,:,i)); end
        
        % [EEGcsp,CSP_features,cspftrs,Psel] = FBCSPfeatures(curr_EEG_cell,EEG.event,1:length(EEG.event),opt);
        [EEGcsp,CSP_features,cspftrs,Psel] = FBCSPfeatures(curr_EEG_cell,trial_data_num,1:length(trial_data_num),fbcsp_opt);
        % convert saveed variables to this! - analyzedData
        save([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_AllEpochs_' 'FBCSP'],'EEGcsp','CSP_features','cspftrs','Psel');% Save the featureeeeeeesssss
        compute_FBCSP = 0;
    end
    
    if compute_BPow [bandPower_features] = EEG_band_powers(EEG.data, EEG.srate); save([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_AllEpochs_' 'BandPowers'],'bandPower_features'); compute_BPow = 0; end
    
    % Compute features over the various sliding window protocols for the various Epochs
    runFeatureComputation = compute_COH + compute_PAC + compute_PLI + compute_dPLI + compute_STE + compute_FBCSP + compute_BPow + compute_Bspec + compute_CFC_SI + compute_PLV;
    if any(runFeatureComputation)
        numFeatures = length(feature_names);
        % Features = cell(1,num_epochs);
        for j = 1:num_epochs
            if runFeatureComputation(j)%%
                disp(['Started processing Epoch ', num2str(j)]);
                if length(size(EEG.data)) > 2
                    dataEpoch = cellfun(@(x)squeeze(x(:,:,j)),EEGdata_filt,'UniformOutput',0);
                    dataEpoch_hilbert = cellfun(@(x)squeeze(x(:,:,j)),EEGdata_filt_hilbert,'UniformOutput',0);
                else
                    dataEpoch = EEGdata_filt; dataEpoch_hilbert = EEGdata_filt_hilbert;
                end
                
                % Initialize variables:
                if compute_COH(j) all_coherence = cell(1,numOfProt); avg_coherence = cell(1,numOfProt); end
                if compute_PAC(j) arr_sortamp = cell(1,numOfProt); arr_plotamp = cell(1,numOfProt); arr_avgamp = cell(1,numOfProt); end
                if compute_PLI(j) PLI = cell(1,numOfProt); PLIcorr = cell(1,numOfProt); PLI_z_score = cell(1,numOfProt); end
                if compute_dPLI(j) dPLI = cell(1,numOfProt); dPLIcorr = cell(1,numOfProt); dPLI_z_score = cell(1,numOfProt); end
                if compute_STE(j) STE = cell(1,numOfProt); NSTE = cell(1,numOfProt); end
                if compute_Bspec(j) Bspec_features_raw_cell = cell(1,numOfProt); Bspec_features_real_cell = cell(1,numOfProt); Bspec_features = cell(1,numOfProt); end
                if compute_CFC_SI(j) CFC_SI = cell(1,numOfProt); CFC_SI_mag = cell(1,numOfProt); CFC_SI_theta = cell(1,numOfProt); end
                
                tic; analyzedData = [];
                for i = 1:numOfProt % Iterate over different windowLength protocols
                    if compute_COH(j) [avg_coherence{i}, all_coherence{i}] = coherence_custom(dataEpoch,windowLength(i),windowStep(i),EEG.srate,par_window); end
                    if compute_PAC(j) [arr_sortamp{i}, arr_plotamp{i}, arr_avgamp{i}] = phase_amp_couple_custom(dataEpoch_hilbert, windowLength(i),windowStep(i),pac_opt,par_window); end
                    if compute_PLI(j) [PLI{i}, PLIcorr{i}, PLI_z_score{i}] = pli_custom(dataEpoch_hilbert,windowLength(i),windowStep(i),pli_opt,par_window); end
                    if compute_dPLI(j) [dPLI{i}, dPLIcorr{i}, dPLI_z_score{i}] = dpli_custom(dataEpoch_hilbert,windowLength(i),windowStep(i),pli_opt,par_window); end
                    if compute_STE(j) [STE{i}, NSTE{i}] = symbolicTransferEntropy(dataEpoch,windowLength(i),windowStep(i)); end
                    if compute_Bspec(j) [Bspec_features_raw_cell{i}, Bspec_features_real_cell{i}, Bspec_features{i}] = bispectrum(dataEpoch,windowLength(i),windowStep(i)); end
                    if compute_CFC_SI(j) [CFC_SI{i}, CFC_SI_mag{i}, CFC_SI_theta{i}] = CFC_SynchronizationIndex(dataEpoch_hilbert, windowLength(i),windowStep(i),CFCSI_opt,par_window); end
                    
                end
                
                if compute_COH(j)
                    if save_all analyzedData.all_coherence = all_coherence; end
                    analyzedData.avg_coherence = avg_coherence;
                end
                if compute_PAC(j)
                    if save_all analyzedData.arr_sortamp = arr_sortamp; end
                    analyzedData.arr_plotamp = arr_plotamp; analyzedData.arr_avgamp = arr_avgamp;
                end
                if compute_PLI(j)
                    if save_all analyzedData.PLI = PLI; analyzedData.PLIcorr = PLIcorr; end
                    analyzedData.PLI_z_score = PLI_z_score;
                end
                if compute_dPLI(j)
                    if save_all analyzedData.dPLI = dPLI; analyzedData.dPLIcorr = dPLIcorr; end
                    analyzedData.dPLI_z_score = dPLI_z_score;
                end
                
                if compute_STE(j) analyzedData.STE = STE; analyzedData.NSTE = NSTE; end
                if compute_Bspec(j) analyzedData.Bspec_features_raw_cell = Bspec_features_raw_cell; analyzedData.Bspec_features_real_cell = Bspec_features_real_cell; analyzedData.Bspec_features = Bspec_features; end
                if compute_CFC_SI(j)
                    if save_all analyzedData.CFC_SI_mag = CFC_SI_mag; analyzedData.CFC_SI_theta = CFC_SI_theta; end
                    analyzedData.CFC_SI = CFC_SI;
                end
                
                disp(['Finished processing Epoch ', num2str(j), ' in ', num2str(toc), 's.']);
                
                % Features{j} = analyzedData;
                
                filename = [curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_Epoch' num2str(j)];
                if already_computed(j) parsave_struct(filename, analyzedData, 1); else parsave_struct(filename, analyzedData, 0); end
                
            else
                disp(['Skipping Epoch ', num2str(j) ' - Already Processed']);
            end
        end
    end
else
    disp(['***************************** Features already computed *****************************']);
end

%% Save all epochs accumulated in Features cell:
% final_filename = [curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_AllEpochs'];
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