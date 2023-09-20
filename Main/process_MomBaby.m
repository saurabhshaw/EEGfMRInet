
% file_ID = 'P1145_Pre';
% file_date = '20190521';
study_name = 'MomBaby';
modality = 'EEG';
dataset_name = '070rs';
session_name = 'rs';

window_step = 10; % in seconds
window_length = 10; % in seconds

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
% base_path_main = fileparts(mfilename('fullpath')); cd(base_path_main); cd ..; % Run this for graham_parfor 
base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
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
base_path_data = [base_path_rd filesep 'MomBabydata_EEG'];
%output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
output_base_path_data = '/project/rrg-beckers/shaws5/Research_data'; %GRAHAM_OUTPUT_PATH

distcomp.feature( 'LocalUseMpiexec', false );
mkdir('/tmp/jobstorage'); %GRAHAM_JOBSTORAGE_LOCATION

%% Process the participants' data:
% dataset_to_use = [file_date '_' study_name '_' modality '-' file_ID];
% dataset_name = file_ID;
chanlocs_file = [base_path_data filesep 'iAverageNet128_v1.sfp'];
elec_transform_file = [base_path filesep 'Cap_files' filesep 'EGI2BrainVision_mod.xlsx'];
model_file = [base_path filesep 'Models' filesep 'SVM_Models' filesep 'individual_Model.mat'];

EEG_data = [];
trial_data = [];
subject_data = [];
session_data = [];

% Create Subject Table/JSON file:
[sub_dir] = dir([base_path_data filesep '*.set']);
tempPID = cellfun(@(x) strsplit(x,{session_name,'.set'}) ,{sub_dir.name},'un',0); tempPID = cellfun(@(x) x{1},tempPID,'un',0); % [sub_dir_mod.PID] = tempPID{:};
tempSID = cellfun(@(x) session_name ,{sub_dir.name},'un',0); % [sub_dir_mod.SID] = tempSID{:};

sub_dir_mod = [];
for i = 1:length(tempPID) 
    sub_dir_mod(i).PID = tempPID{i};
    sub_dir_mod(i).SID = tempSID{i};
end

% sub_dir = [1];
%% For Loop:
sub_imagery_trial_type = [];
sub_EEGdata = [];
jj = 1;

feature_subRange = 3;
%Y_ALL = cell(length(sub_dir),1);
for ii = 137:length(sub_dir) % GRAHAM_PARFOR-1
    dataset_to_use = [sub_dir(ii).name];
    dataset_name = [sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID];
    curr_dir = [base_path_data filesep dataset_name]; %include single participant data-set
    
    %runs_dir = dir([curr_dir filesep '*.xdf']);
    
    %for jj = 1:length(runs_dir)
    try
        fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' num2str(jj) ' ***************************** \n']);
        skip_analysis = 0;
        curr_file = [sub_dir(ii).name];
        
        %% Read in the file:
        EEG = pop_loadset('filename',curr_file,'filepath',base_path_data);
        EEG = pop_chanedit(EEG,'load',{chanlocs_file 'filetype' 'autodetect'});
        EEG.setname = [dataset_name]; EEG = eeg_checkset( EEG );
        
        
        %% Compute and Curate Features:
        % Create output directory if not already made:
        curr_dir = [output_base_path_data filesep dataset_name];
        if isempty(dir(curr_dir)) mkdir(curr_dir); end
        
        % Select and reorder the relevant electrodes:
        egi2brainvision_elec_table = readtable(elec_transform_file,'ReadVariableNames',false);
        egi2brainvision_elec = egi2brainvision_elec_table{:,2};
        if iscell(egi2brainvision_elec) temp = cellfun(@(x)strsplit(x,','),egi2brainvision_elec,'un',0); egi2brainvision_elec = cellfun(@(x)str2num(x{1}),temp); end
        if length(size(EEG.data)) == 2 EEG.data = EEG.data(egi2brainvision_elec,:);
        else EEG.data = EEG.data(egi2brainvision_elec,:,:); end
        
        % Remove any pre-existing windows:
        if length(size(EEG.data)) == 3 EEG = eeg_epoch2continuous(EEG); end
        
        % Add markers for the onset of windows:
        [start_idx, end_idx] = create_windows(size(EEG.data,2), window_step*EEG.srate, window_length*EEG.srate);
        temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
        EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});
        
        % Because some filenames have a ***.*** in them: [NEED TO ADAPT
        % THIS FOR FUTURE RUNS]
        if contains(dataset_name,'.')            
            currFeatures_dir = dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_Epoch*']);
            currFeatures_dir_revised = cellfun(@(x) [replace(x,'.','_') '.mat'],{currFeatures_dir.name},'un',0);
            cellfun(@(x,y) movefile([curr_dir filesep 'EEG_Features' filesep x], [curr_dir filesep 'EEG_Features' filesep y]),{currFeatures_dir.name},currFeatures_dir_revised); 
            dataset_name = replace(dataset_name,'.','_');
        end
        
        % Compute Features:
        currFeatures_dir = dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' dataset_name '_Epoch*.mat']);
        currFeatures_finished = cellfun(@(x) strsplit(x,{'Epoch','.mat'}),{currFeatures_dir.name},'un',0); currFeatures_finished = cellfun(@(x) str2num(x{2}),currFeatures_finished);
        epochs_to_process = setdiff(1:size(EEG.data,3),currFeatures_finished);
        if ~isempty(epochs_to_process)
            %if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
            fprintf(['\n ***************************** Starting Feature Computation ***************************** \n']);
            tic; compute_features_compiled(EEG,curr_dir,dataset_name,feature_names,base_path); toc
        else
            fprintf(['\n ***************************** Features Computed for All Epochs ***************************** \n']);
        end
        
        %         % Compute Features:
        %         if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
        %             fprintf(['***************************** Starting Feature Computation *****************************']);
        %             tic; compute_features_compiled(EEG,curr_dir,dataset_name,feature_names,base_path); toc
        %         end
        
        % Curate features:
        fprintf(['***************************** Curating Computed Features ***************************** \n']);
        Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
        Featurefiles_basename = ['Rev_' dataset_name];
        [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
        
        %          %% Select Features and Classify the data:
        %          run_matfile = 1; % Set this to 1 if 32GB RAM and 0 if 64GB RAM
        %          curr_model = load(model_file);
        %          tic; [final_Features] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile); toc;
        %
        %          [Yhat, Yhat_posterior] = predict_SVM_libsvm(final_Features,curr_model.final_Model,1);
        %
        %          save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'Yhat','Yhat_posterior','model_file');
        %
        
        %% Get features to select:
        if isempty(dir([Featurefiles_directory filesep Featurefiles_basename '_PredictionResults.mat']))
            
            run_matfile = 1; % Set this to 1 if 32GB RAM and 0 if 64GB RAM
            feature_study_name = 'CompositeTask';
            features_to_include = [5];
            
            feature_Featurefiles_directory = [output_base_path_data filesep '20180312_CompositeTask_EEGfMRI-TB' filesep 'Task_block_1' filesep 'EEG_Features'];
            feature_Featurefiles_basename = ['Rev_task_TB_VHDR_TaskBlock1'];
            feature_CONN_cfg = []; feature_CONN_cfg.class_types = 'networks';
            
            % Select the relevant features:
            Featurefiles_curated_dir = dir([feature_Featurefiles_directory filesep feature_Featurefiles_basename '_AllEpochs_*.mat']);
            % Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
            currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);
            
            % Get the number of windows in the learned dataset:
            if run_matfile
                feature_feat_file = matfile([feature_Featurefiles_directory filesep feature_Featurefiles_basename '_AllEpochs_' currFeatures_curated{1} '.mat']);
            else
                feature_feat_file = load([feature_Featurefiles_directory filesep feature_Featurefiles_basename '_AllEpochs_' currFeatures_curated{1} '.mat']);
            end
            feature_curr_feat_size = feature_feat_file.Feature_size;
            feature_num_windows = feature_curr_feat_size(1);
            
            % Get the feature labels from the learned dataset:
            curr_Features_labels = [];
            for i = sort(features_to_include)
                curr_Features_struct = load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' feature_study_name filesep feature_study_name '_' currFeatures_curated{i} '_mRMRiterateGroupResults_' feature_CONN_cfg.class_types],'final_feature_labels_mRMR','final_dataset_mRMR');
                curr_labels = cellfun(@(x)[num2str(i) '_' x],curr_Features_struct.final_feature_labels_mRMR,'un',0);
                curr_Features_labels = cat(2,curr_Features_labels,curr_labels);
            end
            
            % Scale the current number of windows:
            if run_matfile
                feat_file = matfile([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
            else
                feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
            end
            curr_feat_size = feat_file.Feature_size;
            curr_num_windows = curr_feat_size(1);
            
            % Find the mapping from the learned feature labels to the current features:
            feature2curr_start = 1:round(feature_num_windows/curr_num_windows):feature_num_windows;
            if length(feature2curr_start) > curr_num_windows
                feature2curr_cell = arrayfun(@(x)x:(x+round(feature_num_windows/curr_num_windows)-1),feature2curr_start(1:end-1),'un',0);
                leftover_idx = setdiff(feature2curr_start(end-1):feature_num_windows,feature2curr_cell{end});
                feature2curr_cell{end} = [feature2curr_cell{end} leftover_idx];
            else
                feature2curr_cell = arrayfun(@(x)x:(x+round(feature_num_windows/curr_num_windows)-1),feature2curr_start,'un',0);
            end
            
            % Split the feature labels and reconfigure curr_Features_labels:
            all_Feature_labels = cellfun(@(x)strsplit(x,'_'),curr_Features_labels,'un',0);
            feature_window_labels = cellfun(@(x)str2num(x{2}),all_Feature_labels);
            for i = 1:length(feature_window_labels)
                curr_window_labels(i) = find(cellfun(@(x) ismember(feature_window_labels(i),x),feature2curr_cell));
                all_Feature_labels{i}{2} = num2str(curr_window_labels(i));
            end
            curr_Features_labels_mod = cellfun(@(x)strjoin(x,'_'),all_Feature_labels,'un',0);
            
            
            %% Get the selected features for this participant and predict ICN activity:
            variant = 'dPLI';
            curr_model = [];
            curr_model.model_features = currFeatures_curated(features_to_include);
            curr_model.final_feature_labels = curr_Features_labels_mod;
            
            % Get selected features:
            tic; [curr_Features] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile, features_to_include); toc;
            
            % Make ICN predictions:
            confident_threshold = 0.75;
            Yhat = []; Yhat_posterior = []; confident_IDX = [];
            for i = feature_subRange %2:15
                final_saveName = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' feature_study_name filesep 'Classification_Results' filesep 'FinalResults' variant '_' feature_study_name 'LOO' '_FEAT' 'preselected' '_CLASS' feature_CONN_cfg.class_types 'NEW' '_Feat' arrayfun(@(x) num2str(x),features_to_include) '_CVrun' num2str(i)];
                final_Model = load(final_saveName,'Model_SUB'); final_Model = final_Model.Model_SUB{i};
                [Yhat{i}, Yhat_posterior{i}] = predict_SVM_libsvm(curr_Features,final_Model,1);
                confident_IDX{i} = find(sum(Yhat_posterior{i} > confident_threshold,2));
            end
            
            save([Featurefiles_directory filesep Featurefiles_basename '_PredictionResults'],'Yhat','Yhat_posterior','confident_IDX');
            
        else
            disp('**************************** Predictions already completed ***************************');
            load([Featurefiles_directory filesep Featurefiles_basename '_PredictionResults']);
        end
        
        %% Collect the results:
        Yhat_posterior_notempty = cellfun(@(x) ~isempty(x),Yhat_posterior);
        Yhat_posterior_max = []; Yhat_posterior_maxIDX = [];
        for i = feature_subRange [Yhat_posterior_max{i},Yhat_posterior_maxIDX{i}] = max(Yhat_posterior{i},[],2); end
        isempty_Yhat_posterior_max = cellfun(@isempty,Yhat_posterior_max);
        for j = find(isempty_Yhat_posterior_max) Yhat_posterior_max{j} = zeros(size(Yhat{i})); end
        [Yhat_posterior_max_final,Yhat_posterior_max_finalIDX] = max(cell2mat(Yhat_posterior_max),[],2); % Yhat_posterior_maxIDX = cell2mat(Yhat_posterior_maxIDX);
        
        Yhat_maxIDX_final = []; Yhat_final = [];
        for j = 1:length(Yhat_posterior_max_finalIDX)
            Yhat_maxIDX_final(j) = Yhat_posterior_maxIDX{Yhat_posterior_max_finalIDX(j)}(j);
            Yhat_final(j) = Yhat{Yhat_posterior_max_finalIDX(j)}(j);
        end
        
        Yhat_final_SUB{ii,jj} = Yhat_final;
        Yhat_posterior_max_final_SUB{ii,jj} = Yhat_posterior_max_final;
        
        %% Get transistion matrices:
        Y_labels = sort(unique(Yhat_final));
        [orig_mc_full,Y_P_full] = compute_dtmc(Yhat_final,length(Y_labels));
        
        % Create labels:
        state_names = {'CEN', 'DMN', 'SN'};
        Y_P_feature_labels = cell(length(Y_labels));
        for i = 1:length(Y_labels) for j = 1:length(Y_labels) Y_P_feature_labels{i,j} = [state_names{i} '2' state_names{j}]; end; end
        
        Y_P_feature_labels_all = cellfun(@(x)[x],Y_P_feature_labels,'un',0);
        Y_feature_labels = []; Y_feature_labels = cat(2,Y_feature_labels,Y_P_feature_labels_all(:)');
        Y_current = [Y_P_full(:)'];
        
        Y_ALL{ii} = Y_current;
        
%%
         fprintf(['\n ***************************** Finished Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' num2str(jj) ' ***************************** \n']);
         
        
    catch e
        warning(['Problem with Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ]);
        fprintf(1,'The error identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        
        % Write error file:
        failure_string = getReport(e);
        fileID = fopen([curr_dir filesep 'ErrorLog_P' sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID '_Run_' runs_to_include{jj} '.txt'],'w');
        if fileID >= 0 fprintf(fileID,failure_string); fclose(fileID); end
    end
    % end
end % GRAHAM_PARFOR_END

features_to_include = [5];
Results_outputDir = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name];
if isempty(dir(Results_outputDir)) mkdir(Results_outputDir); end
final_saveName = [Results_outputDir filesep 'FinalResults_' study_name '_Feat' arrayfun(@(x) num2str(x),features_to_include)];

save(final_saveName,'Y_ALL','Y_feature_labels','Yhat_final_SUB','Yhat_posterior_max_final_SUB','sub_dir');

% Print results to Excel file:
Y_All_isempty = cellfun(@isempty,Y_ALL');
Y_ALL_test = Y_ALL;
Y_ALL_test(Y_All_isempty) = {nan(1,9)};
Y_ALL_testmat = cell2mat(Y_ALL_test');

%%
% if isempty(dir([base_path_rd filesep 'Analyzed_data' filesep study_name])) mkdir([base_path_rd filesep 'Analyzed_data' filesep study_name]); end
% X = sub_EEGdata; Y = sub_imagery_trial_type;
% curr_dir_data = [base_path_rd filesep 'Analyzed_data' filesep study_name];
% save([curr_dir_data filesep 'Curated_dataset'],'X','Y');
% 
% %% Run feature computation/selection code:
% curr_dir_data = [base_path_rd filesep 'Analyzed_data' filesep study_name];
% load([curr_dir_data filesep 'Curated_dataset.mat']);
% Y_unique = unique(Y);
% TrainAccuracy = zeros(length(Y_unique),length(Y_unique),num_CV_folds); TestAccuracy = zeros(length(Y_unique),length(Y_unique),num_CV_folds); 
% for i = 1:length(Y_unique) % GRAHAM_PARFOR-1
%     for j = 1:length(Y_unique) % GRAHAM_PARFOR-2
%         if i < j
%             Y1 = Y_unique{i}; 
%             Y2 = Y_unique{j};
%             
%             Y1_find = find(cellfun(@(x)strcmp(x,Y1),Y));
%             Y2_find = find(cellfun(@(x)strcmp(x,Y2),Y));
%             
%             YY = [Y1_find Y2_find];
%             
%             XX = X(:,:,YY);
%             YY_final = [zeros(1,length(Y1_find)) ones(1,length(Y2_find))];
%             
%             %%%%%%%%%%% Classifier code runs here %%%%%%%%%%%
%             % Using XX and YY_final
%             
%             % Setup code for feature computation:
%             EEG = []; EEG.data = XX; EEG.srate = srate;            
%             curr_dir = [curr_dir_data filesep Y1 'vs' Y2];
%             if isempty(dir(curr_dir)) mkdir(curr_dir); end 
%             
%             % compute_features_attentionbci
%             if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
%                 fprintf(['***************************** Starting Feature Computation *****************************']);
%                 % dataset_name = 'P1001_Pretrain';
%                 tic; compute_features_compiled(EEG,curr_dir,dataset_name,feature_names,base_path); toc
%             end
%             
%             % Curate features:
%             fprintf(['***************************** Curating Computed Features *****************************']);
%             Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
%             Featurefiles_basename = ['Rev_' dataset_name];
%             [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
% 
%             %% Select Features and Classify the data:
%             fprintf(['***************************** Starting Feature Selection and Classification *****************************']);
% 
%             % Run Feature Selection:
%             nclassesIdx = randperm(length(YY_final));
%             trial_data_num = YY_final(nclassesIdx);
%             [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features(nclassesIdx,:),trial_data_num(nclassesIdx)',max_features);
%             save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR','final_FeatureIDX','Y1','Y2');
%             
%             % Classify:
%             Model = cell(1,num_CV_folds);
%             parfor ii = 1:num_CV_folds
%                 [TrainAccuracy(i,j,ii), TestAccuracy(i,j,ii), Model{ii}] = classify_SVM_libsvm(Features(nclassesIdx,Features_ranked_mRMR),trial_data_num(nclassesIdx)','RBF',testTrainSplit);
%             end
%             
%             save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'TrainAccuracy','TestAccuracy','Model','Features_ranked_mRMR','Features_scores_mRMR','final_FeatureIDX','Y1','Y2');
% 
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%         end
%     end  % GRAHAM_PARFOR_END
% end % GRAHAM_PARFOR_END
% 
% save([curr_dir_data filesep dataset_name '_FullClassificationResults'],'TrainAccuracy','TestAccuracy');

%% Old code:

%         EEG = pop_biosig(curr_file);

% Properly read the events:
%         [~,header] = ReadEDF(curr_file);
%         for kk = 1:length(header.annotation.event) EEG.event(kk).type = header.annotation.event{kk}; end
%         task_conditions = unique(header.annotation.event);

%         % Select the trials that correspond to the classes selected:
%         trial_data_num = trial_data_num(nclassesIdx); Features = Features(nclassesIdx,:);
