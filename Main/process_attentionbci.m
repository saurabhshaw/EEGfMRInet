
% Control Parameters:
runs_to_include = [1]; % From the description of the dataset
dataset_to_use = 'Speaker_audiovisual';
% dataset_name = 'videoSimultaneous_events'; % Run_1 = frontSpeakerAlternating_Events; Run_2 = frontSpeakerSimultaneous_events; Run_1 = videoSimultaneous_events;
dataset_name = 'videoAlternating_events'; %For speaker_audio ; run1 = speaker alternating ; run2 = speaker simultaneous 
nclasses = [2 3]; % all = 1; left = 2; right = 3;
max_features = 1000;%keep this CPU-handle-able
testTrainSplit = 0.75; %classifier - trained on 25%
num_CV_folds = 20; %classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'

% Set Paths:
base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
base_path = pwd;

% Include path:
base_path_data = [base_path filesep 'Data' filesep dataset_to_use];
toolboxes_path = [base_path filesep 'Toolboxes'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB']; happe_directory_path = [toolboxes_path filesep 'Happe']; %HAPPE
addpath(genpath(base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));
addpath(genpath([toolboxes_path filesep 'libsvm-3.23' filesep 'windows'])); % Adding LibSVM
addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

%% Process the participants' data:
EEG_data = [];
trial_data = [];
subject_data = [];
session_data = [];

chanlocs_file = [base_path filesep 'Cap_files' filesep 'Standard-10-20-Cap16.ced'];
% sub_dir = dir([base_path_data filesep 'P*']);
sub_dir = [1];
for ii = 1:length(sub_dir)
    for jj = 1:length(runs_to_include)        
        
        disp(['***************************** Processing Subject ' num2str(ii) ' Session ' num2str(jj) ' *****************************']);
        curr_dir = [base_path_data filesep 'Run' num2str(runs_to_include(jj))]; %include single participant data-set
        curr_filedir = dir([curr_dir filesep '*.set']);
        curr_file = [curr_dir filesep dataset_name '.set'];
        
        %% Read in the file:
        EEG = pop_loadset('filename',[dataset_name '.set'],'filepath',[curr_dir filesep]);
        EEG = pop_chanedit(EEG, 'load',{chanlocs_file 'filetype' 'autodetect'}); EEG = eeg_checkset( EEG );
        EEG.setname = [dataset_name]; EEG = eeg_checkset( EEG );
        
        %% Preprocess datafile if not already preprocessed:
        if isempty(dir([curr_dir filesep 'PreProcessed' filesep dataset_name '_preprocessed.set']))
            % offline_preprocess_HAPPE_attentionbci
            disp(['***************************** Starting Pre-Processing *****************************']);
            offline_preprocess_custom_attentionbci
        else % If data already pre-processed, load it in:
            EEG = pop_loadset('filename',[dataset_name '_preprocessed.set'],'filepath',[curr_dir filesep 'PreProcessed']);
            EEG = eeg_checkset( EEG );
        end
        
        %% Compute EEG features:       
        if isempty(dir([curr_dir filesep 'EEG_Features_Dan' filesep 'Rev_*Epoch*.mat']))
            disp(['***************************** Starting Feature Computation *****************************']);
            compute_features_attentionbci
        end
        
        % Curate Features once computed:
        disp(['***************************** Curating Computed Features *****************************']);
        
        Featurefiles_basename = ['Rev_Sub' num2str(ii) '_Ses' num2str(jj)];
        Featurefiles_directory = [curr_dir filesep 'EEG_Features_Dan'];
        [compute_feat, Features] = curate_features_attentionbci(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
        
        
        %% Select Features and Classify the data:
        disp(['***************************** Starting Feature Selection and Classification *****************************']);

        % Select the trials corresponding to the classes specified above:  
        % trial_data = cellfun(@(x)
        % x{1},{EEG.epoch(:).eventtype},'UniformOutput',0); % Needed for Matlab versions before R2019b
        trial_data = cellfun(@(x) x,{EEG.epoch(:).eventtype},'UniformOutput',0); trial_data_unique = unique(trial_data); % Convert labels to numbers
        % Need to change this for R2019b = automatically take in the trial values, etc.
        [~,trial_data_num] = ismember(trial_data,trial_data_unique); % Convert labels to numbers
        nclassesIdx = []; for i = 1:length(nclasses) nclassesIdx = [nclassesIdx find(trial_data_num == nclasses(i))]; end % Pick the trials corresponding to the relevant classes
        
        % Run Feature Selection:
        [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features(nclassesIdx,:),trial_data_num(nclassesIdx)',max_features);
        
        % Classify:
        TrainAccuracy = zeros(1,num_CV_folds); TestAccuracy = zeros(1,num_CV_folds); Model = cell(1,num_CV_folds);
        parfor i = 1:num_CV_folds
            [TrainAccuracy(i), TestAccuracy(i), Model{i}] = classify_SVM_libsvm(Features(nclassesIdx,Features_ranked_mRMR),trial_data_num(nclassesIdx)','RBF',testTrainSplit);
        end
        
        save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'TrainAccuracy','TestAccuracy','Model','Features_ranked_mRMR','Features_scores_mRMR','nclasses');
        
        disp(['***************************** Finished Subject ' num2str(ii) ' Session ' num2str(jj) ' *****************************']);
    end    
end

%% Old code:

%         EEG = pop_biosig(curr_file);

% Properly read the events:
%         [~,header] = ReadEDF(curr_file);
%         for kk = 1:length(header.annotation.event) EEG.event(kk).type = header.annotation.event{kk}; end
%         task_conditions = unique(header.annotation.event);

%         % Select the trials that correspond to the classes selected:
%         trial_data_num = trial_data_num(nclassesIdx); Features = Features(nclassesIdx,:);
