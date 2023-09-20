
% file_ID = 'P1145_Pre';
% file_date = '20190521';
study_name = 'DatabaseBCI';
modality = 'EEG';
dataset_name = 'P1001_Pretrain';

% Control Parameters:
runs_to_include = {'task'}; % Conditions - From the description of the dataset
seconds_to_cut = 20; srate = 512;
nclasses = [2 3]; % all = 1; left = 2; right = 3;
max_features = 1000;%keep this CPU-handle-able
testTrainSplit = 0.75; %classifier - trained on 25%
num_CV_folds = 20; %classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
% feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
feature_names = {'dPLI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'

% Setup offline_preprocess_cfg:
offline_preprocess_cfg = [];
offline_preprocess_cfg.filter_lp = 0.1; % Was 1 Hz
offline_preprocess_cfg.filter_hp = 50; % Was 40 Hz
offline_preprocess_cfg.segment_data = 1; % 1 for epoched data, 0 for continuous data
offline_preprocess_cfg.segment_markers = {}; % {} is all
offline_preprocess_cfg.task_segment_start = -0.5; % start of the segments in relation to the marker
offline_preprocess_cfg.task_segment_end = 5; % end of the segments in relation to the marker
% offline_preprocess_cfg.max_flatline_duration = 5; max_allowed_jitter = 20; % Parameters required to detect flatline channels
% offline_preprocess_cfg.drift_highpass_band = [0.25 0.75];
offline_preprocess_cfg.ChannelCriterion = 0.8; % To reject channels based on similarity to other channels
offline_preprocess_cfg.run_second_ICA = 0;
offline_preprocess_cfg.save_Workspace = 0;
offline_preprocess_cfg.overwrite_files = 0;
offline_preprocess_cfg.remove_electrodes = 1;
offline_preprocess_cfg.manualICA_check = 0;

% Set Paths:
base_path_main = fileparts(mfilename('fullpath')); cd(base_path_main); cd ..; % Run this for graham_parfor 
% base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
base_path = pwd;

% Include path:
toolboxes_path = [base_path filesep 'Toolboxes']; base_path_main = [base_path filesep 'Main'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB']; happe_directory_path = [toolboxes_path filesep 'Happe']; %HAPPE
addpath(genpath(base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));
addpath(genpath([toolboxes_path filesep 'libsvm-3.23'])); % Adding LibSVM
addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
% addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

[base_path_rc, base_path_rd] = setPaths();
% base_path_rd = ['F:\Saurabh_files\Research_data'];
base_path_data = base_path_rd;
output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
distcomp.feature( 'LocalUseMpiexec', false )

%% Process the participants' data:
% dataset_to_use = [file_date '_' study_name '_' modality '-' file_ID];
% dataset_name = file_ID;
chanlocs_file = [base_path filesep 'Cap_files' filesep 'Biosemi128New_NZ_LPA_RPA.sfp'];

EEG_data = [];
trial_data = [];
subject_data = [];
session_data = [];

% Create Subject Table/JSON file:
[sub_dir,sub_dir_mod] = update_subject_list(study_name,modality,base_path_data,runs_to_include);

% sub_dir = [1];
%% For Loop:
sub_imagery_trial_type = [];
sub_EEGdata = [];

% for ii = 1:length(sub_dir) 
%     dataset_to_use = [sub_dir(ii).name];
%     dataset_name = [sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID];
%     curr_dir = [base_path_data filesep dataset_to_use]; %include single participant data-set
%     
%     runs_dir = dir([curr_dir filesep '*.xdf']);
%     
%     for jj = 1:length(runs_dir)
%         try
%             fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' num2str(jj) ' ***************************** \n']);
%             skip_analysis = 0;
%             curr_file = runs_dir(jj).name;
%             
%             %% Read in the file:
%             EEG = pop_loadxdf([curr_dir filesep curr_file] , 'streamtype', 'EEG', 'exclude_markerstreams', {});
%             EEG = pop_select(EEG, 'nochannel', [1 130:length(EEG.chanlocs)]);
%             EEG = pop_chanedit(EEG, 'load',{chanlocs_file 'filetype' 'autodetect'}); EEG = eeg_checkset( EEG );
%             EEG.setname = [dataset_name 'R' num2str(jj)]; EEG = eeg_checkset( EEG );
%             
%             % Add markers for all the imagery onsets:
%             curr_type = 'Im';
%             offline_preprocess_cfg.task_segment_end = 10; % end of the segments in relation to the marker
%             imagery_events = EEG.event(cellfun(@(x) ~isempty(strfind(x,'Im')),{EEG.event(:).type}));
%             imagery_onset_events = imagery_events(cellfun(@(x) ~isempty(strfind(x,'S')),{imagery_events(:).type}));
%             imagery_trial_type = cellfun(@(x) strsplit(x,{curr_type,'S'}),{imagery_onset_events(:).type},'un',0);
%             imagery_trial_type = cellfun(@(x) x{2},imagery_trial_type,'un',0);
%             for i = 1:length(imagery_onset_events) imagery_onset_events(i).type = [curr_type 'S']; end
%             EEG.event = [EEG.event imagery_onset_events];
%             EEG = eeg_checkset( EEG ,'eventconsistency');
%             
%             offline_preprocess_cfg.segment_markers = {[curr_type 'S']}; % {} is all
%             
%             %% Create output directory if not already made:
%             curr_dir = [output_base_path_data filesep dataset_to_use];
%             if isempty(dir(curr_dir))
%                 mkdir(curr_dir)
%             end
%             
%             %% Preprocess datafile if not already preprocessed:
%             if ~skip_analysis
%                 if isempty(dir([curr_dir filesep 'PreProcessed' filesep EEG.setname '_preprocessed.set']))
%                     % offline_preprocess_HAPPE_attentionbci
%                     fprintf(['\n ***************************** Starting Pre-Processing ***************************** \n']);
%                     tic;
%                     EEG = pop_resample(EEG, srate); EEG.setname = regexprep(EEG.setname,' resampled','');
%                     % EEG = offline_preprocess_manual_deploy(offline_preprocess_cfg,curr_dir,dataset_name,offline_preprocess_cfg.overwrite_files,EEG);
%                     EEG = offline_preprocess_manual_compiled(offline_preprocess_cfg,curr_dir,EEG.setname,offline_preprocess_cfg.overwrite_files,EEG,base_path);
%                     save([curr_dir filesep 'PreProcessed' filesep EEG.setname '_trialType'],'imagery_trial_type');
%                     toc
%                 else % If data already pre-processed, load it in:
%                     EEG = pop_loadset('filename',[EEG.setname '_preprocessed.set'],'filepath',[curr_dir filesep 'PreProcessed']);
%                     EEG = eeg_checkset( EEG );
%                 end
%                 
%                 
%                 %% Process the pre-processed dataset:
%                 curr_trialType = load([curr_dir filesep 'PreProcessed' filesep dataset_name 'R' num2str(jj) '_trialType.mat'],'imagery_trial_type');
%                 curr_segRej = load([curr_dir filesep 'PreProcessed' filesep dataset_name 'R' num2str(jj) '_preprocessed.mat'],'preprocess_segRej');
%                 
%                 sub_imagery_trial_type = cat(2,sub_imagery_trial_type,curr_trialType.imagery_trial_type(~curr_segRej.preprocess_segRej));
%                 
%                 EEG = pop_loadset('filename',[dataset_name 'R' num2str(jj) '_preprocessed.set'],'filepath',[curr_dir filesep 'PreProcessed']);
%                 sub_EEGdata = cat(3,sub_EEGdata,EEG.data);
% 
%                 fprintf(['\n ***************************** Finished Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' num2str(jj) ' ***************************** \n']);
%             else
%                 fprintf(['\n ***************************** Skipping Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' num2str(jj) ' ***************************** \n']);
%                 
%             end
%             
%         catch e
%             warning(['Problem with Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ]);
%             fprintf(1,'The error identifier was:\n%s',e.identifier);
%             fprintf(1,'There was an error! The message was:\n%s',e.message);
%         end
%     end
% end
% 
% if isempty(dir([base_path_rd filesep 'Analyzed_data' filesep study_name])) mkdir([base_path_rd filesep 'Analyzed_data' filesep study_name]); end
% X = sub_EEGdata; Y = sub_imagery_trial_type;
% curr_dir_data = [base_path_rd filesep 'Analyzed_data' filesep study_name];
% save([curr_dir_data filesep 'Curated_dataset'],'X','Y');

%% Set Cluster Properties:
par_window = 1;
if isempty(gcp('nocreate'))
    numCores = feature('numcores')
    p = parpool(numCores);
end

%% Run feature computation/selection code:
curr_dir_data = [base_path_rd filesep 'Analyzed_data' filesep study_name];
load([curr_dir_data filesep 'Curated_dataset.mat']);
Y_unique = unique(Y);
TrainAccuracy = zeros(length(Y_unique),length(Y_unique),num_CV_folds); TestAccuracy = zeros(length(Y_unique),length(Y_unique),num_CV_folds); Model = cell(length(Y_unique),length(Y_unique),num_CV_folds);
for i = 1:length(Y_unique) % GRAHAM_PARFOR-1
    for j = 1:length(Y_unique) % GRAHAM_PARFOR-2
        if i < j
            Y1 = Y_unique{i}; 
            Y2 = Y_unique{j};
            
            Y1_find = find(cellfun(@(x)strcmp(x,Y1),Y));
            Y2_find = find(cellfun(@(x)strcmp(x,Y2),Y));
            
            YY = [Y1_find Y2_find];
            
            XX = X(:,:,YY);
            YY_final = [zeros(1,length(Y1_find)) ones(1,length(Y2_find))];
            
            %%%%%%%%%%% Classifier code runs here %%%%%%%%%%%
            % Using XX and YY_final
            
            % Setup code for feature computation:
            EEG = []; EEG.data = XX; EEG.srate = srate;            
            curr_dir = [curr_dir_data filesep Y1 'vs' Y2];
            if isempty(dir(curr_dir)) mkdir(curr_dir); end 
            
            % compute_features_attentionbci
            if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
                % fprintf(['***************************** Starting Feature Computation ***************************** \n']);
                % dataset_name = 'P1001_Pretrain';
                % tic; compute_features_compiled(EEG,curr_dir,dataset_name,feature_names,base_path); toc
            end
            
            % Curate features:
            % fprintf(['***************************** Curating Computed Features ***************************** \n']);
            Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
            Featurefiles_basename = ['Rev_' dataset_name];
            % [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);

            %% Select Features and Classify the data:
            % fprintf(['***************************** Starting Feature Selection and Classification ***************************** \n']);
            
            fprintf(['************ Processing i = ' num2str(i) '; j = ' num2str(j) ' ************ \n']);
            
            % Run Feature Selection:
            nclassesIdx = randperm(length(YY_final));
            trial_data_num = YY_final(nclassesIdx);
            %[Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features(nclassesIdx,:),trial_data_num(nclassesIdx)',max_features);
            %save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR','final_FeatureIDX','Y1','Y2');
            if ~isempty(dir([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults.mat'])) 
                load([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults.mat']); 
            
            % Classify:
            parfor ii = 1:num_CV_folds
                [TrainAccuracy(i,j,ii), TestAccuracy(i,j,ii), Model{i,j,ii}] = classify_SVM_libsvm(Features(nclassesIdx,Features_ranked_mRMR),trial_data_num(nclassesIdx)','RBF',testTrainSplit);
            end
            
            save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'TrainAccuracy','TestAccuracy','Model','Features_ranked_mRMR','Features_scores_mRMR','final_FeatureIDX','Y1','Y2');
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
    end  % GRAHAM_PARFOR_END
end % GRAHAM_PARFOR_END

save([curr_dir_data filesep dataset_name '_FullClassificationResults'],'TrainAccuracy','TestAccuracy','Model');

%% Old code:

%         EEG = pop_biosig(curr_file);

% Properly read the events:
%         [~,header] = ReadEDF(curr_file);
%         for kk = 1:length(header.annotation.event) EEG.event(kk).type = header.annotation.event{kk}; end
%         task_conditions = unique(header.annotation.event);

%         % Select the trials that correspond to the classes selected:
%         trial_data_num = trial_data_num(nclassesIdx); Features = Features(nclassesIdx,:);
