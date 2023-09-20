
% file_ID = 'P1145_Pre';
% file_date = '20190521';
study_name = 'RunningNFB';
modality = 'EEG';

% Control Parameters:
runs_to_include = {'task','rest'}; % Conditions - From the description of the dataset
seconds_to_cut = 20; srate = 512;
nclasses = [2 3]; % all = 1; left = 2; right = 3;
max_features = 1000; %keep this CPU-handle-able
testTrainSplit = 0.75; %classifier - trained on 25%
num_CV_folds = 20; %classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'

% Setup offline_preprocess_cfg:
offline_preprocess_cfg = [];
offline_preprocess_cfg.filter_lp = 0.1; % Was 1 Hz
offline_preprocess_cfg.filter_hp = 50; % Was 40 Hz
offline_preprocess_cfg.segment_data = 0; % 1 for epoched data, 0 for continuous data
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

% Setup Feature windows:
window_step = 10; % in seconds
window_length = 10; % in seconds

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
% addpath(genpath([toolboxes_path filesep 'libsvm-3.23' filesep 'windows'])); % Adding LibSVM
% addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
% addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

[base_path_rc, base_path_rd] = setPaths();
base_path_data = base_path_rd;
output_base_path_data = '/project/rrg-beckers/shaws5/Research_data'; %GRAHAM_OUTPUT_PATH
temp_filelocation = [output_base_path_data filesep 'temp_deploy_files'];

distcomp.feature( 'LocalUseMpiexec', false );
% mkdir('/tmp/jobstorage'); %GRAHAM_JOBSTORAGE_LOCATION

%% Process the participants' data:
% dataset_to_use = [file_date '_' study_name '_' modality '-' file_ID];
% dataset_name = file_ID;
chanlocs_file = [base_path filesep 'Cap_files' filesep 'Biosemi128New_NZ_LPA_RPA.sfp'];
elec_transform_file = [base_path filesep 'Cap_files' filesep 'Biosemi2BrainVision.xlsx'];
model_file = [base_path filesep 'Models' filesep 'SVM_Models' filesep 'individual_Model.mat'];

EEG_data = [];
trial_data = [];
subject_data = [];
session_data = [];

% Create Subject Table/JSON file:
[sub_dir,sub_dir_mod] = update_subject_list(study_name,modality,base_path_data,runs_to_include);

% Process only those participants that have a pre and a post timepoint:
sub_dir_post_idx = find(cellfun(@(x)contains(x,'_Post'),{sub_dir.name})); % Find all Post files
sub_dir_post_ID = cellfun(@(x) strsplit(x,{'-P','_Post'}) ,{sub_dir(sub_dir_post_idx).name},'un',0); sub_dir_post_ID = cellfun(@(x)str2num(x{2}),sub_dir_post_ID); % Find Post ID numbers
sub_dir_pre_idx_cell = arrayfun(@(y)find(cellfun(@(x)contains(x,[num2str(y) '_Pre']),{sub_dir.name})),sub_dir_post_ID,'un',0); sub_dir_pre_idx_cell_empty = cellfun(@isempty,sub_dir_pre_idx_cell); % Find corresponding Pre files
sub_dir_pre_idx_cell = sub_dir_pre_idx_cell(~sub_dir_pre_idx_cell_empty); % Remove any empty cells
sub_dir_pre_idx = cellfun(@(x)x(length(x)),sub_dir_pre_idx_cell); % In case of multiple Pre scans, keep latest scan
sub_dir_prepost_idx = [sub_dir_pre_idx sub_dir_post_idx(~sub_dir_pre_idx_cell_empty)];

sub_dir = sub_dir(sub_dir_prepost_idx); sub_dir_mod = sub_dir_mod(sub_dir_prepost_idx);
feature_subRange = 2:15;
% sub_dir = [1];
%% For Loop:
Yhat_final_SUB = cell(length(sub_dir),length(runs_to_include));
Yhat_posterior_max_final_SUB = cell(length(sub_dir),length(runs_to_include));

Y_ALL = cell(length(sub_dir),1);
runs_to_include = {'task'};
for ii = 20:20 % GRAHAM_PARFOR
    for jj = 1:length(runs_to_include)        
        try
            tic;
            fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
            skip_analysis = 0;
            dataset_to_use = [sub_dir(ii).name];
            dataset_name = [sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID];
            curr_dir = [base_path_data filesep dataset_to_use]; %include single participant data-set
            
            dataset_date = strsplit(dataset_to_use,'_'); dataset_date = str2num(dataset_date{1});
            
            % This is very specific for this dataset - all the scans after 201905xx had a sampling rate of 512, and all those before were 256:
            if dataset_date < 20190500
                srate = 256;
            else 
                srate = 512;
            end
            %curr_filedir = dir([curr_dir filesep '*.set']);
            %curr_file = [curr_dir filesep dataset_name '.set'];
            
            %% Read in the file:
            switch runs_to_include{jj}
                case 'task'
                    curr_file = [curr_dir filesep 'composite_task_' dataset_name '_full_dataset.mat']; skip_analysis = isempty(dir(curr_file));
                    if ~skip_analysis
                        task_EEG = load(curr_file, 'EEG','class_MARKERS','Exp_blocks');
                        % task_EEG_size = cellfun(@(x)size(x,2),task_EEG.EEG); task_EEG.EEG = cellfun(@(x) x(:,1:min(task_EEG_size)),task_EEG.EEG,'un',0); final_EEG = cat(3,task_EEG.EEG{:}); % Prune all trials to the same size as the smallest one
                        final_EEG = cat(2,task_EEG.EEG{:}); final_EEG_idx = [];
                        for i = 1:length(task_EEG.EEG)
                            final_EEG_idx = [final_EEG_idx repmat(i,[1 size(task_EEG.EEG{i},2)])];
                        end
                    end
                case 'rest'
                    curr_file = [curr_dir filesep 'resting_EEG_' dataset_name '.mat']; skip_analysis = isempty(dir(curr_file));
                    if ~skip_analysis
                        final_EEG = []; final_EEG_idx = [];
                        resting_EEG = load(curr_file, 'resting_EEG'); resting_EEG = resting_EEG.resting_EEG; % Need to do this to nest this inside a parfor loop
                        final_EEG = [final_EEG resting_EEG]; final_EEG_idx = [final_EEG_idx repmat(-1,[1,size(resting_EEG,2)])];
                    end
                case 'combined'
                    curr_file = [curr_dir filesep 'composite_task_' dataset_name '_full_dataset.mat']; skip_analysis = isempty(dir(curr_file));
                    curr_file = [curr_dir filesep 'resting_EEG_' dataset_name '.mat']; skip_analysis = skip_analysis | isempty(dir(curr_file));
                    if ~skip_analysis
                        task_EEG = load([curr_dir filesep 'composite_task_' dataset_name '_full_dataset.mat'], 'EEG','class_MARKERS','Exp_blocks');
                        % task_EEG_size = cellfun(@(x)size(x,2),task_EEG.EEG); task_EEG.EEG = cellfun(@(x) x(:,1:min(task_EEG_size)),task_EEG.EEG,'un',0); final_EEG = cat(3,task_EEG.EEG{:}); % Prune all trials to the same size as the smallest one
                        final_EEG = cat(2,task_EEG.EEG{:}); final_EEG_idx = [];
                        for i = 1:length(task_EEG.EEG)
                            final_EEG_idx = [final_EEG_idx repmat(i,[1 size(task_EEG.EEG{i},2)])];
                        end
                        final_EEG = []; final_EEG_idx = [];
                        resting_EEG = load([curr_dir filesep 'resting_EEG_' dataset_name '.mat'], 'resting_EEG'); final_EEG = [final_EEG resting_EEG]; final_EEG_idx = [final_EEG_idx repmat(-1,[1,size(resting_EEG,2)])];
                    end
            end
            
            if ~skip_analysis
                dataset_name = [runs_to_include{jj} '_' dataset_name];
                EEG = pop_importdata('dataformat','array','nbchan',0,'data','final_EEG','srate',srate,'pnts',0,'xmin',0);
                EEG = pop_chanedit(EEG,'load',{chanlocs_file 'filetype' 'autodetect'});
                % EEG = pop_select(EEG,'time',[seconds_to_cut EEG.xmax-seconds_to_cut]);
                EEG.setname = [dataset_name]; EEG = eeg_checkset( EEG );
            end
                
            %% Create output directory if not already made:
            curr_dir = [output_base_path_data filesep dataset_to_use];
            if isempty(dir(curr_dir))
                mkdir(curr_dir)
            end
            
            if ~skip_analysis
                %% Preprocess datafile if not already preprocessed:
                if isempty(dir([curr_dir filesep 'PreProcessed' filesep dataset_name '_preprocessed.set']))
                    % offline_preprocess_HAPPE_attentionbci
                    fprintf(['\n ***************************** Starting Pre-Processing ***************************** \n']);
                    tic;
                    offline_preprocess_cfg.temp_file = temp_filelocation;
                    % EEG = offline_preprocess_manual_deploy(offline_preprocess_cfg,curr_dir,dataset_name,offline_preprocess_cfg.overwrite_files,EEG);
                    EEG = offline_preprocess_manual_compiled(offline_preprocess_cfg,curr_dir,dataset_name,offline_preprocess_cfg.overwrite_files,EEG,base_path);
                    % offline_preprocess_manual
                    toc
                else % If data already pre-processed, load it in:
                    EEG = pop_loadset('filename',[dataset_name '_preprocessed.set'],'filepath',[curr_dir filesep 'PreProcessed']);
                    EEG = eeg_checkset( EEG );
                end
                
                %% Compute and Curate Features:
                
                % Select and reorder the relevant electrodes:
                biosemi2brainvision_elec_table = readtable(elec_transform_file);
                biosemi2brainvision_elec = biosemi2brainvision_elec_table{:,2};
                if length(size(EEG.data)) == 2 EEG.data = EEG.data(biosemi2brainvision_elec,:);
                else EEG.data = EEG.data(biosemi2brainvision_elec,:,:); end
                
                % Add markers for the onset of windows:
                if length(size(EEG.data)) == 2
                    [start_idx, end_idx] = create_windows(size(EEG.data,2), window_step*EEG.srate, window_length*EEG.srate);
                    temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
                    EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});
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
                
                % Curate features:
                fprintf(['\n ***************************** Curating Computed Features ***************************** \n']);
                Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
                Featurefiles_basename = ['Rev_' dataset_name];
                %[compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
                [compute_feat] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);


                % Select Features and Classify the data:               
%                 curr_model = load(model_file);
%                 tic; [final_Features] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile); toc;
%                 
%                 [Yhat, Yhat_posterior] = predict_SVM_libsvm(final_Features,curr_model.final_Model,1);                
% 
%                 save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'Yhat','Yhat_posterior','model_file');
%                 
                
                

                %% Get features to select:
                if isempty(dir([Featurefiles_directory filesep Featurefiles_basename '_PredictionResults.mat']))
                    
                    run_matfile = 1; % Set this to 1 if 32GB RAM and 0 if 64GB RAM
                    feature_study_name = 'CompositeTask';
                    features_to_include = [1 2 3 4 5];
                    
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
                        feat_file = matfile([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{1} '.mat']);
                    else
                        feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{1} '.mat']);
                    end
                    curr_feat_size = feat_file.Feature_size;
                    curr_num_windows = curr_feat_size(1);
                    
                    % Find the mapping from the learned feature labels to the current features:
                    feature2curr_start = 1:round(feature_num_windows/curr_num_windows):feature_num_windows;
                    feature2curr_cell = arrayfun(@(x)x:(x+round(feature_num_windows/curr_num_windows)-1),feature2curr_start,'un',0);
                    
                    % Split the feature labels and reconfigure curr_Features_labels:
                    all_Feature_labels = cellfun(@(x)strsplit(x,'_'),curr_Features_labels,'un',0);
                    feature_window_labels = cellfun(@(x)str2num(x{2}),all_Feature_labels);
                    for i = 1:length(feature_window_labels)
                        curr_window_labels(i) = find(cellfun(@(x) ismember(feature_window_labels(i),x),feature2curr_cell));
                        all_Feature_labels{i}{2} = num2str(curr_window_labels(i));
                    end
                    curr_Features_labels_mod = cellfun(@(x)strjoin(x,'_'),all_Feature_labels,'un',0);
                    
                    
                    %% Get the selected features for this participant and predict ICN activity:
                    curr_model = [];
                    curr_model.model_features = currFeatures_curated;
                    curr_model.final_feature_labels = curr_Features_labels_mod;
                    
                    % Get selected features:
                    tic; [curr_Features] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile); toc;
                    
                    % Make ICN predictions:
                    confident_threshold = 0.75;
                    Yhat = []; Yhat_posterior = []; confident_IDX = [];
                    for i = feature_subRange
                        final_saveName = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' feature_study_name filesep 'Classification_Results' filesep 'FinalResults_' feature_study_name 'LOO' '_FEAT' 'preselected' '_CLASS' feature_CONN_cfg.class_types 'NEW' '_Feat' arrayfun(@(x) num2str(x),features_to_include) '_CVrun' num2str(i)];
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
                [Yhat_posterior_max_final,Yhat_posterior_max_finalIDX] = max(cell2mat(Yhat_posterior_max),[],2); % Yhat_posterior_maxIDX = cell2mat(Yhat_posterior_maxIDX);
                
                IDX_offset = feature_subRange(1) - 1;
                Yhat_posterior_max_finalIDX = Yhat_posterior_max_finalIDX + IDX_offset;
                
                Yhat_maxIDX_final = []; Yhat_final = [];
                for j = 1:length(Yhat_posterior_max_finalIDX)
                    Yhat_maxIDX_final(j) = Yhat_posterior_maxIDX{Yhat_posterior_max_finalIDX(j)}(j);
                    Yhat_final(j) = Yhat{Yhat_posterior_max_finalIDX(j)}(j);
                end
                
                Yhat_final_SUB{ii,jj} = Yhat_final;
                Yhat_posterior_max_final_SUB{ii,jj} = Yhat_posterior_max_final;
                
                %% Combine with type of WM or AM task blocks (0-AM, 1-WM)
                % Get the task blocks:
                task_block_length = cellfun(@(x) size(x,2),task_EEG.EEG);
                task_block_length_cumsum = cumsum(task_block_length);
                task_block_length_startIDX = []; task_block_length_endIDX = []; task_start_block = []; task_end_block = [];
                for i = 1:length(task_block_length)
                    task_block_length_endIDX(i) = task_block_length_cumsum(i);
                    task_block_length_startIDX(i) = task_block_length_endIDX(i) - task_block_length(i) + 1;
                    
                    % Get the task blocks in terms of feature windows
                    currTask_start_block = find(diff(task_block_length_startIDX(i) < start_idx));
                    currTask_end_block = find(diff(task_block_length_endIDX(i) > end_idx)) + 1;
                    
                    if isempty(currTask_start_block) if (i < length(task_block_length)/2) currTask_start_block = 1; else currTask_start_block = length(start_idx); end; end
                    if isempty(currTask_end_block) if (i < length(task_block_length)/2) currTask_end_block = 1; else currTask_end_block = length(start_idx); end; end

                    task_start_block(i) = currTask_start_block;
                    task_end_block(i) = currTask_end_block;
                end
                
                % Get the mapping to the feature epochs:
                for i = 1:length(start_idx)
                    curr_start_block = find(diff(start_idx(i) > task_block_length_startIDX)); 
                    curr_end_block = find(diff(end_idx(i) < task_block_length_endIDX)) + 1;
                    
                    if isempty(curr_start_block) if (i < length(start_idx)/2) curr_start_block = 1; else curr_start_block = length(task_block_length_startIDX); end; end
                    if isempty(curr_end_block) curr_end_block = 1; end
                    
                    start_block(i) = curr_start_block;
                    end_block(i) = curr_end_block;                    
                end
                
                % Assign the task memberships to the feature epochs:
                task_condition_vect = task_EEG.Exp_blocks(start_block);
                       
                %% Get the percentage of time spent in each trial type:
                state_names = {'CEN', 'DMN', 'SN'};

                Yhat_task = arrayfun(@(x,y)Yhat_final(x:y),task_start_block,task_end_block,'un',0);
                Y_labels = sort(unique(Yhat_final));
                
                Yhat_task_percentage_mean = []; Yhat_task_percentage_mean{1} = []; Yhat_task_percentage_mean{2} = [];
                Yhat_task_percentage_std = []; Yhat_task_percentage_std{1} = []; Yhat_task_percentage_std{2} = [];
                for i = 1:length(Y_labels)
                    
                    % Get overall percentage of time spent:
                    Yhat_task_percentage_mean{1} = [Yhat_task_percentage_mean{1} mean(cellfun(@(x)sum(x==i)/length(x),Yhat_task(task_EEG.Exp_blocks == 0)))];
                    Yhat_task_percentage_mean{2} = [Yhat_task_percentage_mean{2} mean(cellfun(@(x)sum(x==i)/length(x),Yhat_task(task_EEG.Exp_blocks == 1)))];
                    
                    Yhat_task_percentage_std{1} = [Yhat_task_percentage_std{1} std(cellfun(@(x)sum(x==i)/length(x),Yhat_task(task_EEG.Exp_blocks == 0)))];
                    Yhat_task_percentage_std{2} = [Yhat_task_percentage_std{2} std(cellfun(@(x)sum(x==i)/length(x),Yhat_task(task_EEG.Exp_blocks == 1)))];

                end
                
                % Get percentage time active in correct block - DMN in AM blocks and CEN in WM blocks
                Yhat_task_percentage_correct = mean([Yhat_task_percentage_mean{1}(2) Yhat_task_percentage_mean{2}(1)]);
                
                % Gettings this for all blocks:
                orig_mc_taskBlock = []; orig_mcP_taskBlock = []; Y_P_taskBlock = [];
                for i = 1:length(Yhat_task)
                    try
                        [orig_mc_taskBlock{i},Y_P_taskBlock{i}] = compute_dtmc(Yhat_task{i},length(Y_labels));
                    catch
                        [orig_mc_taskBlock{i},Y_P_taskBlock{i}] = compute_dtmc(Yhat_task{i},length(state_names));
                    end
                    orig_mcP_taskBlock{i} = orig_mc_taskBlock{i}.P;
                end
                
                Y_P_taskBlock_mean = []; Y_P_taskBlock_mean{1} = []; Y_P_taskBlock_mean{2} = []; Y_P_taskBlock_std = []; Y_P_taskBlock_std{1} = []; Y_P_taskBlock_std{2} = [];
                Y_P_taskBlock_mean{1} = [Y_P_taskBlock_mean{1} mean(cat(3,Y_P_taskBlock{task_EEG.Exp_blocks == 0}),3)];
                Y_P_taskBlock_mean{2} = [Y_P_taskBlock_mean{2} mean(cat(3,Y_P_taskBlock{task_EEG.Exp_blocks == 1}),3)];
                
                Y_P_taskBlock_std{1} = [Y_P_taskBlock_std{1} std(cat(3,Y_P_taskBlock{task_EEG.Exp_blocks == 0}),[],3)];
                Y_P_taskBlock_std{2} = [Y_P_taskBlock_std{2} std(cat(3,Y_P_taskBlock{task_EEG.Exp_blocks == 1}),[],3)];
                
                try
                    [orig_mc_full,Y_P_full] = compute_dtmc(Yhat_final,length(Y_labels));
                catch
                    [orig_mc_full,Y_P_full] = compute_dtmc(Yhat_final,length(state_names));

                end
                
                % Create labels:
                Y_P_feature_labels = cell(length(Y_labels));
                for i = 1:length(Y_labels) for j = 1:length(Y_labels) Y_P_feature_labels{i,j} = [state_names{i} '2' state_names{j}]; end; end
                
                %% Collect computed features averaged for each window:
                if isempty(dir([Featurefiles_directory filesep 'Blockwise_features*']))
                    disp('**************************** Collecting Blockwise features ***************************');

                    Blockwise_features = cell(1,length(feature_names));
                    for i = 1:length(feature_names)
                        tic
                        currFeatureName = get_finalFeatureName(feature_names(i),featureVar_to_load);
                        currCuratedFeatureFile = [Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatureName{1} '.mat'];
                        
                        currFeatureSize = load(currCuratedFeatureFile,'Feature_size'); currFeatureSize = currFeatureSize.Feature_size;
                        currFeature = load(currCuratedFeatureFile,'Feature');
                        currFeatureBlock = cell(length(currFeature.Feature),1);
                        parfor j = 1:length(currFeatureBlock)
                            currFeatureBlock{j} = mean(reshape(currFeature.Feature{j},currFeatureSize),1);
                        end
                        Blockwise_features{i} = cell2mat(currFeatureBlock);
                        toc
                    end
                    save([Featurefiles_directory filesep 'Blockwise_features'],'Blockwise_features');
                else
                    load([Featurefiles_directory filesep 'Blockwise_features.mat'])
                end
                
                
                %% Plot the results:
                net_idx = mat2cell(1:length(state_names),1,ones(1,length(state_names)));
                title_text = ['Posterior Probability of ICNs'];
                
                load('color_order_shades.mat'); 
                plot_colorord = cell2mat(color_order(:,3)); % Index 3 is the same as the standard Matlab colors
                color_order_idx = [1 2 4]; % The order of colors to use for each of the net_to_analyze
                
                trial_avg_data = mean(cat(3,Yhat_posterior{:}),3); trial_std_data = std(cat(3,Yhat_posterior{:}),[],3)./sqrt(length(Yhat_posterior));
                % x_data = (window_length/2):window_step:(window_length/2)*size(trial_avg_data,1);
                x_data = round(start_idx/EEG.srate) + (window_length/2);
                % h = plot_network_timecourse(x_data,state_names,title_text,net_idx,trial_avg_data,trial_std_data,plot_colorord(color_order_idx,:),'area');
                                
                %% Compute correlation/MI/Cross-correlation between SN and the other networks:
                corr_window = 4; interp_factor = 2;
                MI_lags = [-10:10];
                trial_corr_avg = []; trial_corr_MI = [];
                for curr_net_idx = 1:size(trial_avg_data,2)
                    A = interp(trial_avg_data(:,curr_net_idx),interp_factor); B = interp(trial_avg_data(:,3),interp_factor);
                    if mod(length(A),corr_window) A = A(1:end-mod(length(A),corr_window)); end
                    if mod(length(B),corr_window) B = B(1:end-mod(length(B),corr_window)); end
                    temp = diag(corr(reshape(A,corr_window,[]),reshape(B,corr_window,[])));
                    
                    temp = interp(temp,floor(corr_window./interp_factor));
                    if length(temp) < size(trial_avg_data,1) size_diff = size(trial_avg_data,1) - length(temp); temp = [temp; repmat(temp(end),[size_diff 1])]; 
                    elseif length(temp) > size(trial_avg_data,1) size_diff = length(temp) - size(trial_avg_data,1); temp = temp(1:end-size_diff+1); end
                        
                    trial_corr_avg = cat(2,trial_corr_avg,temp);  
                    
                    % Calculate MI:
                    [v,lag]=ami(A,B,MI_lags);
                end
                trial_corr_std = zeros(size(trial_corr_avg));
                trial_data_Final_avg_plotting = arrayfun(@(x,y) trial_avg_data(x:y,:),task_start_block,task_end_block,'un',0);

                % Resample all trials to fixed length:
                commonLength = 12;
                trial_data_Final_avg_plotting_resamp = cellfun(@(x) resample(x,commonLength,length(x)),trial_data_Final_avg_plotting,'un',0);
                trialResamp_lengths = cellfun(@(x)length(x),trial_data_Final_avg_plotting_resamp);
                trial_data_Final_avg_plotting_resamp = trial_data_Final_avg_plotting_resamp(trialResamp_lengths == commonLength);
                taskBlock_select = task_EEG.Exp_blocks(trialResamp_lengths == commonLength);                
                
                title_text = ['Correlation of ICNs with SN'];
                % h = plot_network_timecourse(x_data,state_names(1:end-1),title_text,net_idx(1:end-1),trial_corr_avg,trial_corr_std,plot_colorord(color_order_idx(1:end-1),:),'area');

                % Calculate the MI:
                trial_MI_Final = []; trial_MIlag_Final = []; trial_MI_Final_max = []; trial_MI_Final_maxIDX = []; trial_MI_Final_maxlag = [];
                trial_MI_Final_firstHalf = []; trial_MIlag_Final_firstHalf = []; trial_MI_Final_max_firstHalf = []; trial_MI_Final_maxIDX_firstHalf = []; trial_MI_Final_maxlag_firstHalf = [];
                trial_MI_Final_secondHalf = []; trial_MIlag_Final_secondHalf = []; trial_MI_Final_max_secondHalf = []; trial_MI_Final_maxIDX_secondHalf = []; trial_MI_Final_maxlag_secondHalf = [];
                for i = 1:length(trial_data_Final_avg_plotting_resamp)
                    x = trial_data_Final_avg_plotting_resamp{i};
                    lag_limit = size(x,1) - 2; % Since the last two values end up being Inf and 0
                    
                    % Get MI and lags:
                    [trial_MI_Final{1}{i},trial_MIlag_Final{1}{i}] = ami(x(:,1),x(:,3),[-lag_limit:lag_limit]);
                    [trial_MI_Final{2}{i},trial_MIlag_Final{2}{i}] = ami(x(:,2),x(:,3),[-lag_limit:lag_limit]);
                    [trial_MI_Final{3}{i},trial_MIlag_Final{3}{i}] = ami(x(:,3),x(:,3),[-lag_limit:lag_limit]);
                    
                    % Find max MI and corresponding lag:
                    for j = 1:length(trial_MI_Final) 
                        [trial_MI_Final_max(i,j),trial_MI_Final_maxIDX(i,j)] = max(trial_MI_Final{j}{i}); 
                        trial_MI_Final_maxlag(i,j) = trial_MIlag_Final{j}{i}(trial_MI_Final_maxIDX(i,j));
                    end                   
                    
                    %% Repeat for firstHalf:
                    x = trial_data_Final_avg_plotting_resamp{i}(1:commonLength/2,:);
                    lag_limit = size(x,1) - 2; % Since the last two values end up being Inf and 0
                    
                    % Get MI and lags:
                    [trial_MI_Final_firstHalf{1}{i},trial_MIlag_Final_firstHalf{1}{i}] = ami(x(:,1),x(:,3),[-lag_limit:lag_limit]);
                    [trial_MI_Final_firstHalf{2}{i},trial_MIlag_Final_firstHalf{2}{i}] = ami(x(:,2),x(:,3),[-lag_limit:lag_limit]);
                    [trial_MI_Final_firstHalf{3}{i},trial_MIlag_Final_firstHalf{3}{i}] = ami(x(:,3),x(:,3),[-lag_limit:lag_limit]);
                    
                    % Find max MI and corresponding lag:
                    for j = 1:length(trial_MI_Final_firstHalf) 
                        [trial_MI_Final_max_firstHalf(i,j),trial_MI_Final_maxIDX_firstHalf(i,j)] = max(trial_MI_Final_firstHalf{j}{i}); 
                        trial_MI_Final_maxlag_firstHalf(i,j) = trial_MIlag_Final_firstHalf{j}{i}(trial_MI_Final_maxIDX_firstHalf(i,j));
                    end 
                    
                    %% Repeat for secondHalf:
                    x = trial_data_Final_avg_plotting_resamp{i}(commonLength/2+1:end,:);
                    lag_limit = size(x,1) - 2; % Since the last two values end up being Inf and 0
                    
                    % Get MI and lags:
                    [trial_MI_Final_secondHalf{1}{i},trial_MIlag_Final_secondHalf{1}{i}] = ami(x(:,1),x(:,3),[-lag_limit:lag_limit]);
                    [trial_MI_Final_secondHalf{2}{i},trial_MIlag_Final_secondHalf{2}{i}] = ami(x(:,2),x(:,3),[-lag_limit:lag_limit]);
                    [trial_MI_Final_secondHalf{3}{i},trial_MIlag_Final_secondHalf{3}{i}] = ami(x(:,3),x(:,3),[-lag_limit:lag_limit]);
                    
                    % Find max MI and corresponding lag:
                    for j = 1:length(trial_MI_Final_secondHalf) 
                        [trial_MI_Final_max_secondHalf(i,j),trial_MI_Final_maxIDX_secondHalf(i,j)] = max(trial_MI_Final_secondHalf{j}{i}); 
                        trial_MI_Final_maxlag_secondHalf(i,j) = trial_MIlag_Final_secondHalf{j}{i}(trial_MI_Final_maxIDX_secondHalf(i,j));
                    end 
                    
                end

                
                % Get the average values for each trial type:
                trial_corr_Final_avg_plotting = arrayfun(@(x,y) trial_corr_avg(x:y,:),task_start_block,task_end_block,'un',0);

                trial_data_Final_avg = arrayfun(@(x,y) mean(trial_avg_data(x:y,:)),task_start_block,task_end_block,'un',0);
                trial_data_Final_std = arrayfun(@(x,y) std(trial_avg_data(x:y,:)),task_start_block,task_end_block,'un',0);

                trial_corr_Final_avg = arrayfun(@(x,y) mean(trial_corr_avg(x:y,:)),task_start_block,task_end_block,'un',0);
                trial_corr_Final_std = arrayfun(@(x,y) std(trial_corr_avg(x:y,:)),task_start_block,task_end_block,'un',0);

                trial_data_Final_mean = []; trial_data_Final_mean{1} = []; trial_data_Final_mean{2} = [];
                trial_data_Final_std = []; trial_data_Final_std{1} = []; trial_data_Final_std{2} = [];
                trial_corr_Final_mean = []; trial_corr_Final_mean{1} = []; trial_corr_Final_mean{2} = [];
                trial_corr_Final_std = []; trial_corr_Final_std{1} = []; trial_corr_Final_std{2} = [];
                
                trial_MI_Final_mean_firstHalf = []; trial_MI_Final_mean_firstHalf{1} = []; trial_MI_Final_mean_firstHalf{2} = [];
                trial_MI_Final_std_firstHalf = []; trial_MI_Final_std_firstHalf{1} = []; trial_MI_Final_std_firstHalf{2} = [];
                trial_MIlag_Final_mean_firstHalf = []; trial_MIlag_Final_mean_firstHalf{1} = []; trial_MIlag_Final_mean_firstHalf{2} = [];
                trial_MIlag_Final_std_firstHalf = []; trial_MIlag_Final_std_firstHalf{1} = []; trial_MIlag_Final_std_firstHalf{2} = [];
                
                trial_MI_Final_mean_secondHalf = []; trial_MI_Final_mean_secondHalf{1} = []; trial_MI_Final_mean_secondHalf{2} = [];
                trial_MI_Final_std_secondHalf = []; trial_MI_Final_std_secondHalf{1} = []; trial_MI_Final_std_secondHalf{2} = [];
                trial_MIlag_Final_mean_secondHalf = []; trial_MIlag_Final_mean_secondHalf{1} = []; trial_MIlag_Final_mean_secondHalf{2} = [];
                trial_MIlag_Final_std_secondHalf = []; trial_MIlag_Final_std_secondHalf{1} = []; trial_MIlag_Final_std_secondHalf{2} = [];
                
                trial_MI_Final_mean = []; trial_MI_Final_mean{1} = []; trial_MI_Final_mean{2} = [];
                trial_MI_Final_std = []; trial_MI_Final_std{1} = []; trial_MI_Final_std{2} = [];
                trial_MIlag_Final_mean = []; trial_MIlag_Final_mean{1} = []; trial_MIlag_Final_mean{2} = [];
                trial_MIlag_Final_std = []; trial_MIlag_Final_std{1} = []; trial_MIlag_Final_std{2} = [];
                
                
                trial_CE_Final = []; trial_CE_Final{1} = []; trial_CE_Final{2} = [];
                trial_CE_Final_mean = []; trial_CE_Final_mean{1} = []; trial_CE_Final_mean{2} = [];
                trial_CE_Final_std = []; trial_CE_Final_std{1} = []; trial_CE_Final_std{2} = [];

                for i = 1:length(Y_labels)
                    % Get activity:
                    trial_data_Final_mean{1} = [trial_data_Final_mean{1} mean(cellfun(@(x)x(i),trial_data_Final_avg(taskBlock_select == 0)))];
                    trial_data_Final_mean{2} = [trial_data_Final_mean{2} mean(cellfun(@(x)x(i),trial_data_Final_avg(taskBlock_select == 1)))];
                    
                    trial_data_Final_std{1} = [trial_data_Final_std{1} std(cellfun(@(x)x(i),trial_data_Final_avg(taskBlock_select == 0)))];
                    trial_data_Final_std{2} = [trial_data_Final_std{2} std(cellfun(@(x)x(i),trial_data_Final_avg(taskBlock_select == 1)))];
                              
                    % Get overall correlations:
                    trial_corr_Final_mean{1} = [trial_corr_Final_mean{1} mean(cellfun(@(x)x(i),trial_corr_Final_avg(taskBlock_select == 0)))];
                    trial_corr_Final_mean{2} = [trial_corr_Final_mean{2} mean(cellfun(@(x)x(i),trial_corr_Final_avg(taskBlock_select == 1)))];
                    
                    trial_corr_Final_std{1} = [trial_corr_Final_std{1} std(cellfun(@(x)x(i),trial_corr_Final_avg(taskBlock_select == 0)))];
                    trial_corr_Final_std{2} = [trial_corr_Final_std{2} std(cellfun(@(x)x(i),trial_corr_Final_avg(taskBlock_select == 1)))];

                    % Get Lagged Mutual Information:
                    trial_MI_Final_mean{1} = [trial_MI_Final_mean{1} mean(trial_MI_Final_max(taskBlock_select == 0,i))];
                    trial_MI_Final_mean{2} = [trial_MI_Final_mean{2} mean(trial_MI_Final_max(taskBlock_select == 1,i))];
                    trial_MIlag_Final_mean{1} = [trial_MIlag_Final_mean{1} mean(trial_MI_Final_maxlag(taskBlock_select == 0,i))];
                    trial_MIlag_Final_mean{2} = [trial_MIlag_Final_mean{2} mean(trial_MI_Final_maxlag(taskBlock_select == 1,i))];

                    trial_MI_Final_std{1} = [trial_MI_Final_std{1} std(trial_MI_Final_max(taskBlock_select == 0,i))];
                    trial_MI_Final_std{2} = [trial_MI_Final_std{2} std(trial_MI_Final_max(taskBlock_select == 1,i))];
                    trial_MIlag_Final_std{1} = [trial_MIlag_Final_std{1} std(trial_MI_Final_maxlag(taskBlock_select == 0,i))];
                    trial_MIlag_Final_std{2} = [trial_MIlag_Final_std{2} std(trial_MI_Final_maxlag(taskBlock_select == 1,i))];

                    trial_MI_Final_mean_firstHalf{1} = [trial_MI_Final_mean_firstHalf{1} mean(trial_MI_Final_max_firstHalf(taskBlock_select == 0,i))];
                    trial_MI_Final_mean_firstHalf{2} = [trial_MI_Final_mean_firstHalf{2} mean(trial_MI_Final_max_firstHalf(taskBlock_select == 1,i))];
                    trial_MIlag_Final_mean_firstHalf{1} = [trial_MIlag_Final_mean_firstHalf{1} mean(trial_MI_Final_maxlag_firstHalf(taskBlock_select == 0,i))];
                    trial_MIlag_Final_mean_firstHalf{2} = [trial_MIlag_Final_mean_firstHalf{2} mean(trial_MI_Final_maxlag_firstHalf(taskBlock_select == 1,i))];

                    trial_MI_Final_std_firstHalf{1} = [trial_MI_Final_std_firstHalf{1} std(trial_MI_Final_max_firstHalf(taskBlock_select == 0,i))];
                    trial_MI_Final_std_firstHalf{2} = [trial_MI_Final_std_firstHalf{2} std(trial_MI_Final_max_firstHalf(taskBlock_select == 1,i))];
                    trial_MIlag_Final_std_firstHalf{1} = [trial_MIlag_Final_std_firstHalf{1} std(trial_MI_Final_maxlag_firstHalf(taskBlock_select == 0,i))];
                    trial_MIlag_Final_std_firstHalf{2} = [trial_MIlag_Final_std_firstHalf{2} std(trial_MI_Final_maxlag_firstHalf(taskBlock_select == 1,i))];

                    trial_MI_Final_mean_secondHalf{1} = [trial_MI_Final_mean_secondHalf{1} mean(trial_MI_Final_max_secondHalf(taskBlock_select == 0,i))];
                    trial_MI_Final_mean_secondHalf{2} = [trial_MI_Final_mean_secondHalf{2} mean(trial_MI_Final_max_secondHalf(taskBlock_select == 1,i))];
                    trial_MIlag_Final_mean_secondHalf{1} = [trial_MIlag_Final_mean_secondHalf{1} mean(trial_MI_Final_maxlag_secondHalf(taskBlock_select == 0,i))];
                    trial_MIlag_Final_mean_secondHalf{2} = [trial_MIlag_Final_mean_secondHalf{2} mean(trial_MI_Final_maxlag_secondHalf(taskBlock_select == 1,i))];

                    trial_MI_Final_std_secondHalf{1} = [trial_MI_Final_std_secondHalf{1} std(trial_MI_Final_max_secondHalf(taskBlock_select == 0,i))];
                    trial_MI_Final_std_secondHalf{2} = [trial_MI_Final_std_secondHalf{2} std(trial_MI_Final_max_secondHalf(taskBlock_select == 1,i))];
                    trial_MIlag_Final_std_secondHalf{1} = [trial_MIlag_Final_std_secondHalf{1} std(trial_MI_Final_maxlag_secondHalf(taskBlock_select == 0,i))];
                    trial_MIlag_Final_std_secondHalf{2} = [trial_MIlag_Final_std_secondHalf{2} std(trial_MI_Final_maxlag_secondHalf(taskBlock_select == 1,i))];
                    
                    % Compute cross-entropy between networks and SN:
                    % cross-entropy( H[p,q]) = -sum(p.log(q)) = H[p] + KLDivergence[p,q]
                    trial_CE_Final{1} = [trial_CE_Final{1} -sum(cell2mat(cellfun(@(x)x(:,i).*log2(x(:,3)),trial_data_Final_avg_plotting_resamp(taskBlock_select == 0),'un',0)),2)];
                    trial_CE_Final{2} = [trial_CE_Final{2} -sum(cell2mat(cellfun(@(x)x(:,i).*log2(x(:,3)),trial_data_Final_avg_plotting_resamp(taskBlock_select == 1),'un',0)),2)];
                end
                
                % Curate the average MI and CE values:
                trial_CE_Final_mean{1} = [mean(trial_CE_Final{1})];
                trial_CE_Final_mean{2} = [mean(trial_CE_Final{2})];                
                trial_CE_Final_std{1} = [std(trial_CE_Final{1})];
                trial_CE_Final_std{2} = [std(trial_CE_Final{2})];
                
                trial_CE_Final_mean_firstHalf{1} = [mean(trial_CE_Final{1}(1:commonLength/2,:))];
                trial_CE_Final_mean_firstHalf{2} = [mean(trial_CE_Final{2}(1:commonLength/2,:))];
                trial_CE_Final_mean_secondHalf{1} = [mean(trial_CE_Final{1}(commonLength/2+1:end,:))];
                trial_CE_Final_mean_secondHalf{2} = [mean(trial_CE_Final{2}(commonLength/2+1:end,:))];
                trial_CE_Final_std_firstHalf{1} = [std(trial_CE_Final{1}(1:commonLength/2,:))];
                trial_CE_Final_std_firstHalf{2} = [std(trial_CE_Final{2}(1:commonLength/2,:))];
                trial_CE_Final_std_secondHalf{1} = [std(trial_CE_Final{1}(commonLength/2+1:end,:))];
                trial_CE_Final_std_secondHalf{2} = [std(trial_CE_Final{2}(commonLength/2+1:end,:))];
               
                % Split by first half and second half of trial:
                trial_data_Final_avg_firstHalf_plotting = arrayfun(@(x,y) trial_avg_data(x:x+floor((y-x)/2),:),task_start_block,task_end_block,'un',0);
                trial_data_Final_avg_secondHalf_plotting = arrayfun(@(x,y) trial_avg_data(y-floor((y-x)/2):y,:),task_start_block,task_end_block,'un',0);
                trial_corr_Final_avg_firstHalf_plotting = arrayfun(@(x,y) trial_corr_avg(x:x+floor((y-x)/2),:),task_start_block,task_end_block,'un',0);
                trial_corr_Final_avg_secondHalf_plotting = arrayfun(@(x,y) trial_corr_avg(y-floor((y-x)/2):y,:),task_start_block,task_end_block,'un',0);
                
                trial_data_Final_avg_firstHalf = arrayfun(@(x,y) mean(trial_avg_data(x:x+floor((y-x)/2),:)),task_start_block,task_end_block,'un',0);
                trial_data_Final_std_firstHalf = arrayfun(@(x,y) std(trial_avg_data(x:x+floor((y-x)/2),:)),task_start_block,task_end_block,'un',0);

                trial_corr_Final_avg_firstHalf = arrayfun(@(x,y) mean(trial_corr_avg(x:x+floor((y-x)/2),:)),task_start_block,task_end_block,'un',0);
                trial_corr_Final_std_firstHalf = arrayfun(@(x,y) std(trial_corr_avg(x:x+floor((y-x)/2),:)),task_start_block,task_end_block,'un',0);

                trial_data_Final_avg_secondHalf = arrayfun(@(x,y) mean(trial_avg_data(y-floor((y-x)/2):y,:)),task_start_block,task_end_block,'un',0);
                trial_data_Final_std_secondHalf = arrayfun(@(x,y) std(trial_avg_data(y-floor((y-x)/2):y,:)),task_start_block,task_end_block,'un',0);

                trial_corr_Final_avg_secondHalf = arrayfun(@(x,y) mean(trial_corr_avg(y-floor((y-x)/2):y,:)),task_start_block,task_end_block,'un',0);
                trial_corr_Final_std_secondHalf = arrayfun(@(x,y) std(trial_corr_avg(y-floor((y-x)/2):y,:)),task_start_block,task_end_block,'un',0);
                
                trial_data_Final_mean_firstHalf = []; trial_data_Final_mean_firstHalf{1} = []; trial_data_Final_mean_firstHalf{2} = [];
                trial_data_Final_std_firstHalf = []; trial_data_Final_std_firstHalf{1} = []; trial_data_Final_std_firstHalf{2} = [];
                trial_corr_Final_mean_firstHalf = []; trial_corr_Final_mean_firstHalf{1} = []; trial_corr_Final_mean_firstHalf{2} = [];
                trial_corr_Final_std_firstHalf = []; trial_corr_Final_std_firstHalf{1} = []; trial_corr_Final_std_firstHalf{2} = [];
                
                trial_data_Final_mean_secondHalf = []; trial_data_Final_mean_secondHalf{1} = []; trial_data_Final_mean_secondHalf{2} = [];
                trial_data_Final_std_secondHalf = []; trial_data_Final_std_secondHalf{1} = []; trial_data_Final_std_secondHalf{2} = [];
                trial_corr_Final_mean_secondHalf = []; trial_corr_Final_mean_secondHalf{1} = []; trial_corr_Final_mean_secondHalf{2} = [];
                trial_corr_Final_std_secondHalf = []; trial_corr_Final_std_secondHalf{1} = []; trial_corr_Final_std_secondHalf{2} = [];
                for i = 1:length(Y_labels)
                    % Get activity:
                    trial_data_Final_mean_firstHalf{1} = [trial_data_Final_mean_firstHalf{1} mean(cellfun(@(x)x(i),trial_data_Final_avg_firstHalf(taskBlock_select == 0)))];
                    trial_data_Final_mean_firstHalf{2} = [trial_data_Final_mean_firstHalf{2} mean(cellfun(@(x)x(i),trial_data_Final_avg_firstHalf(taskBlock_select == 1)))];
                    
                    trial_data_Final_std_firstHalf{1} = [trial_data_Final_std_firstHalf{1} std(cellfun(@(x)x(i),trial_data_Final_avg_firstHalf(taskBlock_select == 0)))];
                    trial_data_Final_std_firstHalf{2} = [trial_data_Final_std_firstHalf{2} std(cellfun(@(x)x(i),trial_data_Final_avg_firstHalf(taskBlock_select == 1)))];
                              
                    % Get overall correlations:
                    trial_corr_Final_mean_firstHalf{1} = [trial_corr_Final_mean_firstHalf{1} mean(cellfun(@(x)x(i),trial_corr_Final_avg_firstHalf(taskBlock_select == 0)))];
                    trial_corr_Final_mean_firstHalf{2} = [trial_corr_Final_mean_firstHalf{2} mean(cellfun(@(x)x(i),trial_corr_Final_avg_firstHalf(taskBlock_select == 1)))];
                    
                    trial_corr_Final_std_firstHalf{1} = [trial_corr_Final_std_firstHalf{1} std(cellfun(@(x)x(i),trial_corr_Final_avg_firstHalf(taskBlock_select == 0)))];
                    trial_corr_Final_std_firstHalf{2} = [trial_corr_Final_std_firstHalf{2} std(cellfun(@(x)x(i),trial_corr_Final_avg_firstHalf(taskBlock_select == 1)))];

                    % Get activity:
                    trial_data_Final_mean_secondHalf{1} = [trial_data_Final_mean_secondHalf{1} mean(cellfun(@(x)x(i),trial_data_Final_avg_secondHalf(taskBlock_select == 0)))];
                    trial_data_Final_mean_secondHalf{2} = [trial_data_Final_mean_secondHalf{2} mean(cellfun(@(x)x(i),trial_data_Final_avg_secondHalf(taskBlock_select == 1)))];
                    
                    trial_data_Final_std_secondHalf{1} = [trial_data_Final_std_secondHalf{1} std(cellfun(@(x)x(i),trial_data_Final_avg_secondHalf(taskBlock_select == 0)))];
                    trial_data_Final_std_secondHalf{2} = [trial_data_Final_std_secondHalf{2} std(cellfun(@(x)x(i),trial_data_Final_avg_secondHalf(taskBlock_select == 1)))];
                              
                    % Get overall correlations:
                    trial_corr_Final_mean_secondHalf{1} = [trial_corr_Final_mean_secondHalf{1} mean(cellfun(@(x)x(i),trial_corr_Final_avg_secondHalf(taskBlock_select == 0)))];
                    trial_corr_Final_mean_secondHalf{2} = [trial_corr_Final_mean_secondHalf{2} mean(cellfun(@(x)x(i),trial_corr_Final_avg_secondHalf(taskBlock_select == 1)))];
                    
                    trial_corr_Final_std_secondHalf{1} = [trial_corr_Final_std_secondHalf{1} std(cellfun(@(x)x(i),trial_corr_Final_avg_secondHalf(taskBlock_select == 0)))];
                    trial_corr_Final_std_secondHalf{2} = [trial_corr_Final_std_secondHalf{2} std(cellfun(@(x)x(i),trial_corr_Final_avg_secondHalf(taskBlock_select == 1)))];

                end
                
                %% Accumulate the predictors:
                Y_P_feature_labels_AM = cellfun(@(x)['AM_' x],Y_P_feature_labels,'un',0);
                Y_P_feature_labels_WM = cellfun(@(x)['WM_' x],Y_P_feature_labels,'un',0);
                Y_P_feature_labels_all = cellfun(@(x)['all_' x],Y_P_feature_labels,'un',0);

                Y_feature_labels = {'AM_CEN','AM_DMN','AM_SN','WM_CEN','WM_DMN','WM_SN','Correct_ICN'};
                Y_feature_labels = cat(2,Y_feature_labels,Y_P_feature_labels_AM(:)');
                Y_feature_labels = cat(2,Y_feature_labels,Y_P_feature_labels_WM(:)');                
                Y_feature_labels = cat(2,Y_feature_labels,Y_P_feature_labels_all(:)');
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_Yhat','AM_DMN_Yhat','AM_SN_Yhat','WM_CEN_Yhat','WM_DMN_Yhat','WM_SN_Yhat'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_corrSN','AM_DMN_corrSN','AM_SN_corrSN','WM_CEN_corrSN','WM_DMN_corrSN','WM_SN_corrSN'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_Yhat_1half','AM_DMN_Yhat_1half','AM_SN_Yhat_1half','WM_CEN_Yhat_1half','WM_DMN_Yhat_1half','WM_SN_Yhat_1half'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_corrSN_1half','AM_DMN_corrSN_1half','AM_SN_corrSN_1half','WM_CEN_corrSN_1half','WM_DMN_corrSN_1half','WM_SN_corrSN_1half'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_Yhat_2half','AM_DMN_Yhat_2half','AM_SN_Yhat_2half','WM_CEN_Yhat_2half','WM_DMN_Yhat_2half','WM_SN_Yhat_2half'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_corrSN_2half','AM_DMN_corrSN_2half','AM_SN_corrSN_2half','WM_CEN_corrSN_2half','WM_DMN_corrSN_2half','WM_SN_corrSN_2half'});
                
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_miSN','AM_DMN_miSN','AM_SN_miSN','WM_CEN_miSN','WM_DMN_miSN','WM_SN_miSN'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_miSN_1half','AM_DMN_miSN_1half','AM_SN_miSN_1half','WM_CEN_miSN_1half','WM_DMN_miSN_1half','WM_SN_miSN_1half'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_miSN_2half','AM_DMN_miSN_2half','AM_SN_miSN_2half','WM_CEN_miSN_2half','WM_DMN_miSN_2half','WM_SN_miSN_2half'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_ceSN','AM_DMN_ceSN','AM_SN_ceSN','WM_CEN_ceSN','WM_DMN_ceSN','WM_SN_ceSN'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_ceSN_1half','AM_DMN_ceSN_1half','AM_SN_ceSN_1half','WM_CEN_ceSN_1half','WM_DMN_ceSN_1half','WM_SN_ceSN_1half'});
                Y_feature_labels = cat(2,Y_feature_labels,{'AM_CEN_ceSN_2half','AM_DMN_ceSN_2half','AM_SN_ceSN_2half','WM_CEN_ceSN_2half','WM_DMN_ceSN_2half','WM_SN_ceSN_2half'});

                
                Y_current = [Yhat_task_percentage_mean{1} Yhat_task_percentage_mean{2} Yhat_task_percentage_correct,...
                    Y_P_taskBlock_mean{1}(:)' Y_P_taskBlock_mean{2}(:)' Y_P_full(:)'];
                
                Y_current = [Y_current trial_data_Final_mean{1} trial_data_Final_mean{2} trial_corr_Final_mean{1} trial_corr_Final_mean{2}];
                Y_current = [Y_current trial_data_Final_mean_firstHalf{1} trial_data_Final_mean_firstHalf{2} trial_corr_Final_mean_firstHalf{1} trial_corr_Final_mean_firstHalf{2}];
                Y_current = [Y_current trial_data_Final_mean_secondHalf{1} trial_data_Final_mean_secondHalf{2} trial_corr_Final_mean_secondHalf{1} trial_corr_Final_mean_secondHalf{2}];

                Y_current = [Y_current trial_MI_Final_mean{1} trial_MI_Final_mean{2}];
                Y_current = [Y_current trial_MI_Final_mean_firstHalf{1} trial_MI_Final_mean_firstHalf{2}];
                Y_current = [Y_current trial_MI_Final_mean_secondHalf{1} trial_MI_Final_mean_secondHalf{2}];

                Y_current = [Y_current trial_CE_Final_mean{1} trial_CE_Final_mean{2}];
                Y_current = [Y_current trial_CE_Final_mean_firstHalf{1} trial_CE_Final_mean_firstHalf{2}];
                Y_current = [Y_current trial_CE_Final_mean_secondHalf{1} trial_CE_Final_mean_secondHalf{2}];

                
                Y_ALL{ii} = Y_current;
                
                toc
                %% Use the learned model to predict class labels:
                %A = unique(cat(1,[],confident_IDX{:}));
%                 X = curr_Features_classify(testIdx_SUB,Features_ranked_mRMR_SUB);
%                 Y = curr_YY_final_classify(testIdx_SUB);
%                 
%                 X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));
%                 
%                 [YTesthat, testaccur, YTesthat_posterior] = svmpredict(Y', X_scaled, Model_SUB{current_test_block},' -b 1');
%                 
%                 % Isolate the time points that were confident (high class probability):
%                 picked_IDX = find(sum(YTesthat_posterior > 0.75,2));
%                 TestAccuracy_confident = sum(YTesthat(picked_IDX) == Y(picked_IDX)')/length(picked_IDX);
%                 
%                 % Use the confident time points for individualized mRMR
%                 [~,curr_labels_mRMR_subIDX2curr_subIDX] = find(trial_select_bin);
%                 origSpace_select_testIDX = curr_labels_mRMR_subIDX2curr_subIDX(select_testIDX);
%                 optimal_testIDX = origSpace_select_testIDX(picked_IDX);
%                 % curate_features_mRMR_compiled([Featurefiles_basename '_CLASS' CONN_cfg.class_types], Featurefiles_directory, YY_final, max_features, task_dir, base_path)
%                 
%                 % Use the classification probabilties as weights for weighted-mRMR
%                 % feature selection:
%                 weights = max(YTesthat_posterior,[],2);
%                 
%                 %% Give optimal_testIDX as curr_dataset_mRMR_IDX to run mRMR on this custom
%                 % dataset
%                 name_suffix = ['_IND' sub_dir_mod(current_test_block).PID '_weighted'];
%                 %     if isempty(dir([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_' currFeatures_curated{1} '_mRMRiterateGroupResults_' CONN_cfg.class_types name_suffix '.mat']))
%                 %         for i = features_to_include  
%                 %             curate_features_mRMR_group_LOOP_deploy(i,Featurefiles_directory_ALL,curr_Featurefiles_basename_ALL,YTesthat(picked_IDX),name_suffix,output_base_path_data,study_name,max_features,CONN_cfg,optimal_testIDX)
%                 %             curate_features_mRMR_group_LOOP_weighted_deploy(i,Featurefiles_directory_ALL,curr_Featurefiles_basename_ALL,YTesthat,weights,[name_suffix '_weighted'],output_base_path_data,study_name,max_features,CONN_cfg,origSpace_select_testIDX)
%                 %             curate_features_mRMR_group_LOOP_weighted_matlab_deploy(i,Featurefiles_directory_ALL,curr_Featurefiles_basename_ALL,YTesthat,weights,[name_suffix '_weighted_matlab'],output_base_path_data,study_name,max_features,CONN_cfg,origSpace_select_testIDX)
%                 %         end 
%                 %     end
%                 
%                 %% Load the features from this customized run:
%                 INDcurr_labels_mRMR = YTesthat(picked_IDX);
%                 INDcurr_Features = []; INDcurr_Features_labels = [];
%                 for i = 1:length(currFeatures_curated)
%                     INDcurr_Features_struct = load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'LeaveOneOut_IND_Features' filesep study_name '_' currFeatures_curated{i} '_mRMRiterateGroupResults_' CONN_cfg.class_types name_suffix],'final_feature_labels_mRMR','final_dataset_mRMR');
%                     %INDcurr_Feature_vect = nan(length(INDcurr_labels_mRMR),size(INDcurr_Features_struct.final_dataset_mRMR,2));
%                     %INDcurr_IDX = squeeze(INDcurr_Features_struct.curr_dataset_mRMR_IDX);
%                     
%                     %INDcurr_Feature_vect(INDcurr_IDX,:) = INDcurr_Features_struct.final_dataset_mRMR;
%                     INDcurr_labels = cellfun(@(x)[num2str(i) '_' x],INDcurr_Features_struct.final_feature_labels_mRMR,'un',0);
%                     
%                     INDcurr_Features = cat(2,INDcurr_Features,INDcurr_Features_struct.final_dataset_mRMR);
%                     INDcurr_Features_labels = cat(2,INDcurr_Features_labels,INDcurr_labels);
%                     
%                 end
%                 
%                 % Identify all the sessions for this participant:
%                 curr_sess_IDX = find(subIDX_ALL == current_test_block);
%                 
%                 %% Get the selected features for this participant:
%                 curr_model = [];
%                 curr_model.model_features = currFeatures_curated;
%                 curr_model.final_feature_labels = INDcurr_Features_labels;
%                 IND_Features = cell(1,length(curr_sess_IDX));
%                 tic
%                 parfor i = 1:length(curr_sess_IDX)
%                     tic; [IND_Features{i}] = get_selected_features(curr_model, Featurefiles_basename_ALL{curr_sess_IDX(i)}, Featurefiles_directory_ALL{curr_sess_IDX(i)}, run_matfile); toc;
%                 end
%                 toc
%                 
%                 %% Train a custom model using the pre-classified results as training samples:
%                 origSpace_curr_subIDX = curr_labels_mRMR_subIDX == current_test_block;
%                 curr_trial_select_bin = trial_select_bin(find(origSpace_curr_subIDX));
%                 origSpace_curr_sessIDX = curr_labels_mRMR_sessIDX(find(origSpace_curr_subIDX));
%                 sessIDX_unique = unique(origSpace_curr_sessIDX);
%                 sess_curr_trial_select_bin = cell(1,length(sessIDX_unique));
%                 for i = 1:length(sessIDX_unique)
%                     sess_curr_trial_select_bin{i} = find(curr_trial_select_bin(origSpace_curr_sessIDX == sessIDX_unique(i)));
%                 end
%                 
%                 IND_Features_selected = cellfun(@(x,y) x(y,:),IND_Features,sess_curr_trial_select_bin,'un',0);
%                 IND_Features_all = cat(1,[],IND_Features_selected{:});
%                 
%                 IND_YY_final_selected = cellfun(@(x,y) x(y),YY_final_ALL(curr_sess_IDX)',sess_curr_trial_select_bin,'un',0);
%                 IND_YY_Final_all = cat(2,[],IND_YY_final_selected{:});
%                 
%                 trainIdx_IND = picked_IDX; testIdx_IND = setdiff([1:length(IND_YY_Final_all)],picked_IDX);
%                 IND_YY_Final_all(trainIdx_IND) = YTesthat(picked_IDX)'; % Replace the original labels with that derived from the general model
%                 
%                 [IND_trial_select_bin,IND_class_weights] = fix_classImbalance(IND_YY_Final_all(trainIdx_IND),'balance',0);
%                 
%                 IND_Features_all_isNOTnan = sum(isnan(IND_Features_all)) == 0;
%                 IND_Features_all = IND_Features_all(:,IND_Features_all_isNOTnan);
%                 % Run last Feature Selection:
%                 num_top_feat = 250;
%                 if final_featureSelection
%                     if isempty(dir([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_FeatureSelectionGroupResults_' CONN_cfg.class_types name_suffix]))
%                         %[Features_ranked_mRMR_IND, Features_scores_mRMR_IND] = mRMR_custom(IND_Features_all,IND_YY_Final_all',max_features*3);
%                         [Features_ranked_mRMR_IND, Features_scores_mRMR_IND] = mRMR_custom(IND_Features_all,YTesthat',max_features*3);
%                         save([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_FeatureSelectionGroupResults_' CONN_cfg.class_types name_suffix],'Features_ranked_mRMR_IND','Features_scores_mRMR_IND');
%                     else
%                         load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_FeatureSelectionGroupResults_' CONN_cfg.class_types name_suffix]);
%                     end
%                     Features_ranked_mRMR_IND = Features_ranked_mRMR_IND(1:num_top_feat);
%                 else
%                     Features_ranked_mRMR_IND = 1:size(IND_Features_all,2);
%                     Features_scores_mRMR_IND = nan(1,size(IND_Features_all,2));
%                 end
%                 
%                 %[TrainAccuracy_IND(current_test_block), TestAccuracy_IND(current_test_block), Model_IND{current_test_block}] = classify_SVMweighted_libsvm(IND_Features_all,YTesthat,'RBF',0,IND_class_weights,trainIdx_IND,testIdx_IND);
%                 [TrainAccuracy_IND(current_test_block), TestAccuracy_IND(current_test_block), Model_IND{current_test_block}] = classify_SVMweighted_libsvm(IND_Features_all(:,Features_ranked_mRMR_IND),YTesthat,'RBF',0,class_weights,trainIdx_IND,testIdx_IND);
%                 
%                 % Check accuracy:
%                 X = IND_Features_all(testIdx_IND,:);
%                 Y = IND_YY_Final_all(testIdx_IND);
%                 
%                 X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));
%                 
%                 [IND_YTesthat, IND_testaccur, IND_YTesthat_posterior] = svmpredict(Y', X_scaled, Model_IND{current_test_block},' -b 1');
%                 
%                 
                
                
                fprintf(['\n ***************************** Finished Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
            else
                fprintf(['\n ***************************** Skipping Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
                
            end
            
        catch e
            warning(['Problem with Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ]);
            fprintf(1,'\n The error identifier was:\n%s',e.identifier);
            fprintf(1,'\n There was an error! The message was:\n%s',e.message);
            
            % Write error file:
            failure_string = getReport(e);
            fileID = fopen(['ErrorLog_P' sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID '_Run_' runs_to_include{jj} '.txt'],'w');
            fprintf(fileID,failure_string); fclose(fileID);
        end
    end
end % GRAHAM_PARFOR_END

features_to_include = [1 2 3 4 5];
Results_outputDir = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name];
if isempty(dir(Results_outputDir)) mkdir(Results_outputDir); end
final_saveName = [Results_outputDir filesep 'FinalResults_' study_name '_Feat' arrayfun(@(x) num2str(x),features_to_include)];

Y_ALL(cellfun(@isempty,Y_ALL)) = {ones(1,length(Y_feature_labels))*(-1)}; % Make all the empty ones -1.

save(final_saveName,'Y_ALL','Y_feature_labels','Yhat_final_SUB','Yhat_posterior_max_final_SUB','sub_dir');

%% Old code:

%         EEG = pop_biosig(curr_file);

% Properly read the events:
%         [~,header] = ReadEDF(curr_file);
%         for kk = 1:length(header.annotation.event) EEG.event(kk).type = header.annotation.event{kk}; end
%         task_conditions = unique(header.annotation.event);

%         % Select the trials that correspond to the classes selected:
%         trial_data_num = trial_data_num(nclassesIdx); Features = Features(nclassesIdx,:);
