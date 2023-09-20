
% file_ID = 'P1145_Pre';
% file_date = '20190521';
study_name = 'CompositeTask';
modality = 'EEGfMRI';

% Study specific parameters:
study_conditions = {'ABM','WM','Rest','ABM-WM','WM-ABM'};
replace_files = 0;
scan_parameters = [];
scan_parameters.TR = 2; % MRI Repetition Time (in seconds)
scan_parameters.anat_num_images = 184;
scan_parameters.rsfunc_num_images = 5850;
scan_parameters.tfunc_num_images = 11700;
scan_parameters.slicespervolume = 39;
scan_parameters.slice_marker = 'R128';
scan_parameters.ECG_channel = 32;
scan_parameters.srate = 5000;
scan_parameters.low_srate = 500;

% Control Parameters:
runs_to_include = {'task'}; % From the description of the dataset can be 'task', 'rest' or 'combined'
overwrite_files = 0;
srate = 5000;
% nclasses = [2 3]; % all = 1; left = 2; right = 3;
nclasses = [0:6]; % all = 1; left = 2; right = 3;
max_features = 1000; % Keep this CPU-handle-able
testTrainSplit = 0.75; % Classifier - trained on 25%
num_CV_folds = 20; % Classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'
final_featureSelection = 1; % Whether to run final feature selection after mRMR iterate - NOTE - this caused poor classification results

% Setup CONN_cfg:
CONN_cfg = [];
CONN_cfg.CONN_analysis_name = 'ROI'; % The name of the CONN first-level analysis
% CONN_cfg.CONN_analysis_name = 'V2V_02'; % The name of the CONN first-level analysis
% CONN_cfg.CONN_project_name = 'conn_composite_task_fMRI'; % The name of the CONN project
CONN_cfg.CONN_project_name = 'conn_composite_task_fMRI_corrected'; % The name of the CONN project
CONN_cfg.CONN_data_type = 'ICA'; % The source of the CONN data - can be ICA or ROI 
CONN_cfg.net_to_analyze = {'CEN', 'DMN', 'SN'}; % List all networks to analyze
CONN_cfg.use_All_cond = 1; % If this is 1, use the timecourses from condition 'All'
CONN_cfg.p_norm = 0; % Lp norm to use to normalize the timeseries data:  For data normalization - set to 0 to skip it, 1 = L1 norm and 2 = L2 norm
CONN_cfg.conditions_to_include = [1 2]; % The condition indices to sum up in the norm
CONN_cfg.window_step = 2; % in seconds - Used for Feature windows:
CONN_cfg.window_length = 10; % in seconds - Used for Feature windows:
CONN_cfg.threshold = 0.3; % Threshold for making the time course binary
CONN_cfg.class_types = 'subnetworks'; % Can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze
CONN_cfg.multilabel = 0; % Whether classification is multilabel or single label
CONN_cfg.ROIs_toUse = {'ICA_CEN','ICA_LCEN','ICA_anteriorSN','ICA_posteriorSN','ICA_ventralDMN','ICA_dorsalDMN'}; % Need this if using ROIs for labels rather than ICA
CONN_cfg.rescale = 1; % Rescale between 0 and 1 if selected

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
offline_preprocess_cfg.preMRItruncation = 1; % Number of seconds to keep before the start of the MRI slice markers

% Setup Feature windows:
% window_step = 2; % in seconds
% window_length = 10; % in seconds

% Set Paths:
% base_path_main = fileparts(mfilename('fullpath')); cd(base_path_main); cd ..;
% base_path = pwd;
base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
base_path = pwd;

% Include path:
toolboxes_path = [base_path filesep 'Toolboxes']; base_path_main = [base_path filesep 'Main'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB']; happe_directory_path = [toolboxes_path filesep 'Happe']; %HAPPE
addpath(genpath(base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));
addpath(genpath([toolboxes_path filesep 'libsvm-3.23'])); % Adding LibSVM
% addpath(genpath([toolboxes_path filesep 'libsvm-3.23' filesep 'windows'])); % Adding LibSVM
addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
addpath(genpath([toolboxes_path filesep 'Mricron'])) % Adding Mricron
% addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

% Setup Data paths:
[base_path_rc, base_path_rd] = setPaths();
base_path_data = base_path_rd;
output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
offline_preprocess_cfg.temp_file = [output_base_path_data filesep 'temp_deploy_files'];

distcomp.feature( 'LocalUseMpiexec', false );
% mkdir('/tmp/jobstorage'); %GRAHAM_JOBSTORAGE_LOCATION

% Setup CONN psths:
% This is the folder that has the data from the CONN project (in CONN file structure) 
% [conn_folder filesep CONN_project_name filesep 'results' filesep 'firstlevel' filesep CONN_analysis_name];
% And the main project .mat file in the main CONN project folder
CONN_cfg.CONNresults_folder = [base_path filesep 'Models' filesep 'ICA_Labels' filesep CONN_cfg.CONN_project_name '-' CONN_cfg.CONN_analysis_name];
warning off

%% Process the participants' data:
% dataset_to_use = [file_date '_' study_name '_' modality '-' file_ID];
% dataset_name = file_ID;
% study_prefix = 'composite_task_';
temp_study_name = find(isstrprop(study_name,'upper')); study_base_name = '';
for i = 1:length(temp_study_name)
    str_start_idx = temp_study_name(i); 
    if (i+1) > length(temp_study_name) str_end_idx = length(study_name); else str_end_idx = temp_study_name(i+1) - 1; end
    if i == 1 study_base_name = lower(study_name(str_start_idx:str_end_idx)); else study_base_name = [study_base_name '_' lower(study_name(str_start_idx:str_end_idx))]; end
end
study_prefix = [study_base_name '_'];

chanlocs_file = [base_path filesep 'Cap_files' filesep 'BrainProductsMR64_NZ_LPA_RPA_fixed.sfp'];

% Create Subject Table/JSON file:
% [sub_dir,sub_dir_mod] = conn_addSubjects_EEGfMRI(study_name,modality,base_path_data, scan_parameters,study_conditions, replace_files);
% [sub_dir,sub_dir_mod] = conn_addSubjects_corrections_EEGfMRI(study_name,modality,base_path_data, scan_parameters,study_conditions,chanlocs_file,replace_files);
[sub_dir,sub_dir_mod] = update_subject_list(study_name,modality,base_path_data,runs_to_include);

%% Get Label data:
CONN_data = conn_loadROI_data(CONN_cfg,scan_parameters);
% CONN_data = conn_loadICA_data(CONN_cfg,scan_parameters);
EEGfMRI_corr = readtable([base_path_rd filesep 'EEGfMRI_ProcessingCorrespondence_' study_name '.xls']); % This is to make sure that the blocks of the EEG and fMRI correspond together
EEGfMRI_corr_firstblockIDX = find(cellfun(@(x) strcmp(x,'EEGTaskBlocks'),EEGfMRI_corr.Properties.VariableNames));
EEGfMRI_corr_validIDX = cellfun(@(x)~isnan(x),table2cell(EEGfMRI_corr(:,1))); EEGfMRI_corr = EEGfMRI_corr(EEGfMRI_corr_validIDX,:);
EEGfMRI_corrIDX = zeros(sum(EEGfMRI_corr_validIDX),(size(EEGfMRI_corr,2)-EEGfMRI_corr_firstblockIDX+1));
for i = 1:size(EEGfMRI_corr,1)
    temp_corrIDX = table2cell(EEGfMRI_corr(i,EEGfMRI_corr_firstblockIDX:size(EEGfMRI_corr,2)));
    for j = 1:length(temp_corrIDX)
        if ischar(temp_corrIDX{j}) temp_corrIDX{j} = str2num(temp_corrIDX{j}); end
        if isempty(temp_corrIDX{j}) temp_corrIDX{j} = NaN; end
    end
    EEGfMRI_corrIDX(i,:) = cell2mat(temp_corrIDX);
end
%% Run Loop:
Feature_labels_split_LOO = [];
Feature_labels_split_LOO.window = [];
Feature_labels_split_LOO.freqBand_A = [];
Feature_labels_split_LOO.freqBand_B = [];
Feature_labels_split_LOO.chan_A = [];
Feature_labels_split_LOO.chan_B = [];
Feature_labels_split_LOO.PID = [];
Feature_labels_split_LOO.SID = [];

Feature_labels_string_LOO = [];

for ii = 2:length(sub_dir) % GRAHAM_PARFOR-1
    for jj = 1:length(runs_to_include)
        %%
        try
            fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
            skip_analysis = 0;
            dataset_to_use = [sub_dir(ii).name];
            dataset_name = [sub_dir_mod(ii).PID];
            curr_dir = [base_path_data filesep dataset_to_use]; %include single participant data-set

            %% Read in the file:
            switch runs_to_include{jj}
                case 'task'
                    task_DIR = dir([curr_dir filesep 'Task_block_*']);
                   
                case 'rest'
                    task_DIR = [1];              
            end
                        
            %% Create output directory if not already made:
            curr_dir = [output_base_path_data filesep dataset_to_use];
            if isempty(dir(curr_dir))
                mkdir(curr_dir)
            end
            
            %% Run analysis:
            Feature_labels_split = [];
            Feature_labels_split.feature = [];
            Feature_labels_split.window = [];
            Feature_labels_split.freqBand_A = [];
            Feature_labels_split.freqBand_B = [];
            Feature_labels_split.chan_A = [];
            Feature_labels_split.chan_B = [];
            for m = 1:length(task_DIR) % GRAHAM_PARFOR-2
                
                % Check if this task block has been processed by CONN:
                if ~isnan(EEGfMRI_corrIDX(ii,m))
                    
                    % Obtain the label vector:
                    curr_CONN_IDX = EEGfMRI_corrIDX(ii,m);
                    YY_final = cell2mat(CONN_data.fMRI_labels_selected_window_avg_thresh{ii-1}{curr_CONN_IDX});  % NOTE:only because this is excluding the first subject
                    YY_final_continuous = (CONN_data.fMRI_labels_selected_window_avg{ii-1}{curr_CONN_IDX}); YY_final_continuous = cat(1,YY_final_continuous{:}); % NOTE:only because this is excluding the first subject
                    
                    % Select relevant features:
                    % nclassesIdx = randperm(length(YY_final));
                    % [Features,Feature_labels_mRMR,Feature_mRMR_order] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features);
                    % save([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults'],'Features','Feature_labels_mRMR','Feature_mRMR_order');
                    task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
                    Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
                    Featurefiles_basename = ['Rev_' curr_dataset_name];
                    
                    if ~isempty(dir([task_dir filesep 'EEG_Features' filesep '*_mRMRiterateResults.mat']))
                        % curate_features_mRMR_compiled(Featurefiles_basename, Featurefiles_directory, YY_final, max_features, task_dir, base_path)
                        
                        mRMRiterateResults = load([Featurefiles_directory filesep Featurefiles_basename '_CLASS' CONN_cfg.class_types '_mRMRiterateResults.mat']);
                        
                        Feature_labels = cellfun(@(x)strsplit(x,'_'),mRMRiterateResults.final_feature_labels_mRMR,'un',0);
                        for i = 1:length(Feature_labels)
                            if mod(i,max_features) == 0
                                curr_Feat = floor(i/max_features);
                            else
                                curr_Feat = floor(i/max_features)+1;
                            end
                            Feature_labels_split.feature{i} = mRMRiterateResults.currFeatures_curated{curr_Feat};
                            Feature_labels_split.window(m,i) = str2num(Feature_labels{i}{1});
                            Feature_labels_split.freqBand_A(m,i) = str2num(Feature_labels{i}{2});
                            if length(Feature_labels{i}) > 4
                                Feature_labels_split.freqBand_B(m,i) = str2num(Feature_labels{i}{3});
                                Feature_labels_split.chan_A(m,i) = str2num(Feature_labels{i}{4});
                                Feature_labels_split.chan_B(m,i) = str2num(Feature_labels{i}{5});
                            else
                                Feature_labels_split.freqBand_B(m,i) = NaN;
                                Feature_labels_split.chan_A(m,i) = str2num(Feature_labels{i}{3});
                                Feature_labels_split.chan_B(m,i) = str2num(Feature_labels{i}{4});
                            end
                        end
                        
                        % Accumulate the Feature Labels:
                        Feature_labels_split_LOO.PID = cat(1,Feature_labels_split_LOO.PID,ii);
                        Feature_labels_split_LOO.SID = cat(1,Feature_labels_split_LOO.SID,m);
                        Feature_labels_split_LOO.feature = Feature_labels_split.feature;
                        
                        % Accumulate the string Feeature Labels:
                        Feature_labels_string_LOO = cat(1,Feature_labels_string_LOO,mRMRiterateResults.final_feature_labels_mRMR);
                        
                        %                             if final_featureSelection
                        %                                 if isempty(dir([task_dir filesep 'EEG_Features' filesep '*_FeatureSelectionResults.mat']))
                        %                                     [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features,YY_final',max_features);
                        %                                     save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR');
                        %                                 else
                        %                                     load([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults.mat']);
                        %                                 end
                        %                                 % [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features,YY_final',max_features);
                        %                                 % save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR');
                        %                             else
                        %                                 Features_ranked_mRMR = 1:size(Features,2);
                        %                                 Features_scores_mRMR = nan(1,size(Features,2));
                        %                             end
                        %
                        %                             % Fix class imbalance and select Features and index:
                        %                             % [trial_select_bin] = fix_classImbalance(YY_final,'balance',0);
                        %                             [trial_select_bin,class_weights] = fix_classImbalance(YY_final,'balance',0);
                        %                             Features = Features(find(trial_select_bin),:); YY_final = YY_final(find(trial_select_bin));
                        %
                        %                             % Run Classification:
                        %                             Model = cell(1,num_CV_folds); Model_null = cell(1,num_CV_folds);
                        %                             parfor kk = 1:num_CV_folds % Make this par
                        %                                 % Run regular classification:
                        %                                 [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
                        %                                 % [TrainAccuracy(m,kk), TestAccuracy(m,kk), Model{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),YY_final','RBF',0,trainIdx{kk},testIdx{kk});
                        %                                 [TrainAccuracy(m,kk), TestAccuracy(m,kk), Model{kk}] = classify_SVMweighted_libsvm(Features,YY_final','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
                        %
                        %                                 % Run null classification (after randomly permuting the labels of the testing trials):
                        %                                 Y_null_train = YY_final(trainIdx{kk}); Y_null = YY_final;
                        %                                 Y_null(trainIdx{kk}) = Y_null_train(randperm(length(Y_null_train)));
                        %                                 % [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
                        %                                 [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVMweighted_libsvm(Features,Y_null','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
                        %                             end
                        %
                        %                             save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'TrainAccuracy','TestAccuracy','Model','TrainAccuracy_null','TestAccuracy_null','Model_null','testIdx','trainIdx');
                        %
                        %                             % Accumulate Features for between task classification:
                        %                             Features_SUB{m} = Features;
                        %                             Feature_labels_mRMR_SUB{m} = Feature_labels_mRMR_SUB;
                        %                             YY_final_SUB{m} = YY_final;
                        %                             YY_final_continuous_SUB{m} = YY_final_continuous;
                        %
                        %                             % Run Regression:
                        %                             YY_final_continuous = YY_final_continuous(find(trial_select_bin),:);
                        %                             Reg_Model = cell(num_CV_folds,size(YY_final_continuous,2)); Reg_Model_null = cell(num_CV_folds,size(YY_final_continuous,2));
                        %                             for kk = 1:num_CV_folds % Make this par
                        %                                 [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
                        %                                 for tt = 1:size(YY_final_continuous,2)
                        %                                     [Reg_TrainAccuracy(m,kk,tt), Reg_TestAccuracy(m,kk,tt), Reg_Model{kk,tt}] = classify_RSVM_matlab(Features,YY_final_continuous(:,tt),'RBF',0,trainIdx{kk},testIdx{kk});
                        %                                 end
                        %                             end
                        
                        fprintf(['\n ***************************** Finished Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ' num2str(m) ' ***************************** \n']);
                    else
                        fprintf(['\n ***************************** Skipping Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ' num2str(m) ' ***************************** \n']);
                        
                    end
                end
            end
            

            Feature_labels_split_LOO.window = cat(1,Feature_labels_split_LOO.window,Feature_labels_split.window);
            Feature_labels_split_LOO.freqBand_A = cat(1,Feature_labels_split_LOO.freqBand_A,Feature_labels_split.freqBand_A);
            Feature_labels_split_LOO.freqBand_B = cat(1,Feature_labels_split_LOO.freqBand_B,Feature_labels_split.freqBand_B);
            Feature_labels_split_LOO.chan_A = cat(1,Feature_labels_split_LOO.chan_A,Feature_labels_split.chan_A);
            Feature_labels_split_LOO.chan_B = cat(1,Feature_labels_split_LOO.chan_B,Feature_labels_split.chan_B);
            
            
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

%% Find common features to be used for LOO classification:
final_Feature_labels_LOO = []; test_sub_IDX = 2;
unique_features = unique(Feature_labels_split_LOO.feature);
num_rows = size(Feature_labels_split_LOO.window,1);

final_type = 'string_labels'; % Can be 'freq' or 'common' or 'string_labels'
final_Feature_labels_LOO = []; final_Feature_labels_string_LOO = [];

final_Feature_sizes = cellfun(@(x) load([output_base_path_data filesep sub_dir(test_sub_IDX).name filesep 'Task_block_1' filesep 'EEG_Features' filesep 'Rev_task_' sub_dir_mod(test_sub_IDX).PID '_VHDR_TaskBlock1_AllEpochs_' x '.mat'],'Feature_size'),unique_features,'un',0);
final_Feature_sizes = cellfun(@(x) x.Feature_size ,final_Feature_sizes,'un',0);
final_Feature_labels_freq_LOO = cellfun(@(x) arrayfun(@(y)zeros(1,y),x,'un',0),final_Feature_sizes,'un',0);
final_Feature_labels_freqSUB_LOO = cellfun(@(x) arrayfun(@(y)zeros(num_rows,y),x,'un',0),final_Feature_sizes,'un',0);

% final_Feature_labels_freqSUB_LOO = [];
% final_Feature_labels_freqSUB_LOO.window = [];
% final_Feature_labels_freqSUB_LOO.freqBand_A = [];
% final_Feature_labels_freqSUB_LOO.freqBand_B = [];
% final_Feature_labels_freqSUB_LOO.chan_A = [];
% final_Feature_labels_freqSUB_LOO.chan_B = [];

for j = 1:length(unique_features)
    curr_feature_IDX = cellfun(@(x)strcmp(x,unique_features{j}),Feature_labels_split_LOO.feature);
    
    switch final_type
        case 'common'
            
            final_Feature_labels_LOO.window{j} = Feature_labels_split_LOO.window(1,curr_feature_IDX);
            final_Feature_labels_LOO.freqBand_A{j} = Feature_labels_split_LOO.freqBand_A(1,curr_feature_IDX);
            final_Feature_labels_LOO.freqBand_B{j} = Feature_labels_split_LOO.freqBand_B(1,curr_feature_IDX);
            final_Feature_labels_LOO.chan_A{j} = Feature_labels_split_LOO.chan_A(1,curr_feature_IDX);
            final_Feature_labels_LOO.chan_B{j} = Feature_labels_split_LOO.chan_B(1,curr_feature_IDX);
            
            for i = 2:size(Feature_labels_split_LOO.window,1)
                if isempty(find(Feature_labels_split_LOO.window(i,curr_feature_IDX) == 0))
                    final_Feature_labels_LOO.window{j} = intersect(final_Feature_labels_LOO.window{j},Feature_labels_split_LOO.window(i,curr_feature_IDX));
                    final_Feature_labels_LOO.freqBand_A{j} = intersect(final_Feature_labels_LOO.freqBand_A{j},Feature_labels_split_LOO.freqBand_A(i,curr_feature_IDX));
                    final_Feature_labels_LOO.freqBand_B{j} = intersect(final_Feature_labels_LOO.freqBand_B{j},Feature_labels_split_LOO.freqBand_B(i,curr_feature_IDX));
                    final_Feature_labels_LOO.chan_A{j} = intersect(final_Feature_labels_LOO.chan_A{j},Feature_labels_split_LOO.chan_A(i,curr_feature_IDX));
                    final_Feature_labels_LOO.chan_B{j} = intersect(final_Feature_labels_LOO.chan_B{j},Feature_labels_split_LOO.chan_B(i,curr_feature_IDX));
                end
            end
            
        case 'freq'
            for i = 1:size(Feature_labels_split_LOO.window,1)
                curr_num_feats = size(Feature_labels_split_LOO.window(i,curr_feature_IDX),2);
                unique_window = unique(Feature_labels_split_LOO.window(i,curr_feature_IDX));
                if unique_window ~= 0
                    final_Feature_labels_freqSUB_LOO{j}{1}(i,unique_window) = arrayfun(@(x)sum(Feature_labels_split_LOO.window(i,curr_feature_IDX) == x)./curr_num_feats,unique_window);
                    
                    unique_freqBand_A = unique(Feature_labels_split_LOO.freqBand_A(i,curr_feature_IDX));
                    final_Feature_labels_freqSUB_LOO{j}{2}(i,unique_freqBand_A) = arrayfun(@(x)sum(Feature_labels_split_LOO.freqBand_A(i,curr_feature_IDX) == x)./curr_num_feats,unique_freqBand_A);
                    
                    if strcmp(unique_features{j},'CFC_SI_mag') || strcmp(unique_features{j},'arr_avgamp')
                        unique_freqBand_B = unique(Feature_labels_split_LOO.freqBand_B(i,curr_feature_IDX));
                        final_Feature_labels_freqSUB_LOO{j}{3}(i,unique_freqBand_B) = arrayfun(@(x)sum(Feature_labels_split_LOO.freqBand_B(i,curr_feature_IDX) == x)./curr_num_feats,unique_freqBand_B);
                    end
                    
                    unique_chan_A = unique(Feature_labels_split_LOO.chan_A(i,curr_feature_IDX));
                    final_Feature_labels_freqSUB_LOO{j}{end-1}(i,unique_chan_A) = arrayfun(@(x)sum(Feature_labels_split_LOO.chan_A(i,curr_feature_IDX) == x)./curr_num_feats,unique_chan_A);
                    
                    unique_chan_B = unique(Feature_labels_split_LOO.chan_B(i,curr_feature_IDX));
                    final_Feature_labels_freqSUB_LOO{j}{end}(i,unique_chan_B) = arrayfun(@(x)sum(Feature_labels_split_LOO.chan_B(i,curr_feature_IDX) == x)./curr_num_feats,unique_chan_B);
                end
            end
            
            for m = 1:length(final_Feature_labels_freqSUB_LOO{j})
                final_Feature_labels_freq_LOO{j}{m}(:) = sum(final_Feature_labels_freqSUB_LOO{j}{m},1)./size(final_Feature_labels_freqSUB_LOO{j}{m},1);
            end
            
            
        case 'string_labels'
            %             final_Feature_labels_string_LOO{j} = Feature_labels_string_LOO(1,curr_feature_IDX);
            %
            %             for i = 2:size(Feature_labels_string_LOO,1)
            %                 final_Feature_labels_string_LOO{j} = intersect(final_Feature_labels_string_LOO{j},Feature_labels_string_LOO(i,curr_feature_IDX));
            %             end
            curr_num_feats = size(Feature_labels_string_LOO(1,curr_feature_IDX),2);

            curr_Features_string = cellfun(@(x) strsplit(x,'_') ,Feature_labels_string_LOO(:,curr_feature_IDX),'un',0);
            for mm = 1:size(curr_Features_string,1)
                for kk = 1:size(curr_Features_string,2)
                    temp_string = curr_Features_string{mm,kk};
                    final_string = temp_string{2};
                    for ll = 3:length(temp_string) % Skip the windows
                        final_string = [final_string '_' temp_string{ll}];
                    end
                    curr_Features_string{mm,kk} = final_string;
                end
            end
            % curr_Features_string = cellfun(@(x) strsplit(x,'_') ,curr_Features_string,'un',0);

            final_Feature_labels_string_LOO{j}.unique_features = unique(Feature_labels_string_LOO(:,curr_feature_IDX)); % If want to include windows
            % final_Feature_labels_string_LOO{j}.unique_features = unique(curr_Features_string); % If want to remove windows

            curr_unique_feats_freq = zeros(size(final_Feature_labels_string_LOO{j}.unique_features));
            curr_unique_feats = final_Feature_labels_string_LOO{j}.unique_features;
            parfor i = 1:length(final_Feature_labels_string_LOO{j}.unique_features)
                x = curr_unique_feats{i};
                curr_unique_feats_freq(i) = sum(cellfun(@(y) strcmp(x,y),Feature_labels_string_LOO(:,curr_feature_IDX)),'All');
                % curr_unique_feats_freq(i) = sum(cellfun(@(y) strcmp(x,y),curr_Features_string),'All');
                disp([num2str(i) ' / ' num2str(length(final_Feature_labels_string_LOO{j}.unique_features))]);
            end
            final_Feature_labels_string_LOO{j}.unique_features_freq = curr_unique_feats_freq;
            % cellfun(@(x) sum(cellfun(@(y) strcmp(x,y),Feature_labels_string_LOO(:,curr_feature_IDX))),final_Feature_labels_string_LOO{j}.unique_features,'un',0)
    end
end

%% Sort and select final feature indices:
% Specify percentage of the top features within each feature to keep - 
top_percentage = 0.02;
final_feature_labels = [];
for j = 1:length(unique_features)
    curr_feature_IDX = cellfun(@(x)strcmp(x,unique_features{j}),Feature_labels_split_LOO.feature);
    
    switch final_type
        case 'common'
            
        case 'freq'
  
        case 'string_labels'
            
            curr_top_num = floor(top_percentage*length(final_Feature_labels_string_LOO{j}.unique_features));
            
            [curr_feat_freq_sort,curr_feat_freq_sortIDX] = sort(final_Feature_labels_string_LOO{j}.unique_features_freq,'descend');
            
            final_Feature_labels_string_LOO{j}.selected_features = final_Feature_labels_string_LOO{j}.unique_features(curr_feat_freq_sortIDX(1:curr_top_num));
            curr_feature_labels = cellfun(@(x)[num2str(j) '_' x],final_Feature_labels_string_LOO{j}.selected_features,'un',0);
            final_feature_labels = cat(1,final_feature_labels,curr_feature_labels);
    end
end

%% Accumulate feature indices:
group_features = [];
group_features.feature = [];
group_features.window = [];
group_features.freqBand_A = [];
group_features.freqBand_B = [];
group_features.chan_A = [];
group_features.chan_B = [];

for j = 1:length(unique_features)
    curr_feat = unique_features{j};
    for k = 1:length(final_Feature_labels_LOO.window{j})
        curr_window = final_Feature_labels_LOO.window{j}(k);
        
        for l = 1:length(final_Feature_labels_LOO.freqBand_A{j})
            curr_freqBand_A = final_Feature_labels_LOO.freqBand_A{j}(l);

            if ~isempty(final_Feature_labels_LOO.freqBand_B{j}) || strcmp(curr_feat,'CFC_SI_mag') || strcmp(curr_feat,'arr_avgamp')
                for m = 1:length(final_Feature_labels_LOO.freqBand_B{j})
                    curr_freqBand_B = final_Feature_labels_LOO.freqBand_B{j}(m);
                    for n = 1:length(final_Feature_labels_LOO.chan_A{j})
                        curr_chan_A = final_Feature_labels_LOO.chan_A{j}(n);
                        for o = 1:length(final_Feature_labels_LOO.chan_B{j})
                            curr_chan_B = final_Feature_labels_LOO.chan_B{j}(o);
                            
                            group_features.feature = cat(2,group_features.feature,curr_feat);
                            group_features.window = cat(2,group_features.window,curr_window);
                            group_features.freqBand_A = cat(2,group_features.freqBand_A,curr_freqBand_A);
                            group_features.freqBand_B = cat(2,group_features.freqBand_B,curr_freqBand_B);
                            group_features.chan_A = cat(2,group_features.chan_A,curr_chan_A);
                            group_features.chan_B = cat(2,group_features.chan_B,curr_chan_B);
                            
                        end
                    end
                end
                
            else                
                for n = 1:length(final_Feature_labels_LOO.chan_A{j})
                    curr_chan_A = final_Feature_labels_LOO.chan_A{j}(n);
                    for o = 1:length(final_Feature_labels_LOO.chan_B{j})
                        curr_chan_B = final_Feature_labels_LOO.chan_B{j}(o);
                        
                        group_features.feature = cat(2,group_features.feature,j);
                        group_features.window = cat(2,group_features.window,curr_window);
                        group_features.freqBand_A = cat(2,group_features.freqBand_A,curr_freqBand_A);
                        group_features.freqBand_B = cat(2,group_features.freqBand_B,NaN);
                        group_features.chan_A = cat(2,group_features.chan_A,curr_chan_A);
                        group_features.chan_B = cat(2,group_features.chan_B,curr_chan_B);
                        
                        
                        
                    end
                end
            end
        end
    end
end

%% Run Classification between participants, between task blocks:
% % Run Feature Selection:
% curr_Features = cat(1,Features_SUB{:}); curr_YY_final = cat(2,YY_final_SUB{:}); curr_YY_final_continuous = cat(1,YY_final_continuous_SUB{:});
% 
% Featurefiles_basename = ['Rev_' runs_to_include{jj} '_' dataset_name '_VHDR'];
% 
% if final_featureSelection
%     [Features_ranked_mRMR_SUB, Features_scores_mRMR_SUB] = mRMR(curr_Features,curr_YY_final',max_features*3);
%     save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults_SUB'],'Features_SUB','Features_ranked_mRMR_SUB','Feature_labels_mRMR_SUB','Features_scores_mRMR_SUB');
% else
%     Features_ranked_mRMR_SUB = 1:size(curr_Features,2);
%     Features_scores_mRMR_SUB = nan(1,size(curr_Features,2));
% end
% 
% % Fix class imbalance and select Features and index:
% [trial_select_bin,class_weights] = fix_classImbalance(curr_YY_final,'balance',0);
% curr_Features = curr_Features(find(trial_select_bin),:); curr_YY_final = curr_YY_final(find(trial_select_bin));
% 
% % Run Classification:
% Model_SUB = cell(1,num_CV_folds); Model_null_SUB = cell(1,num_CV_folds);
% for kk = 1:num_CV_folds % make this par
%     % Run regular classification:
%     % Identify the block to keep as the testing data:
%     current_test_block{kk} = randi(length(task_DIR));
%     current_train_blocks{kk} = setdiff([1:length(task_DIR)],current_test_block{kk});
%     
%     % Create dataset accordingly:
%     curr_Features_test = cat(1,Features_SUB{current_test_block{kk}}); curr_YY_final_test = cat(2,YY_final_SUB{current_test_block{kk}});
%     curr_Features_train = cat(1,Features_SUB{current_train_blocks{kk}}); curr_YY_final_train = cat(2,YY_final_SUB{current_train_blocks{kk}});
%     
%     curr_Features = [curr_Features_test; curr_Features_train]; curr_YY_final = [curr_YY_final_test curr_YY_final_train];
%     testIdx_SUB{kk} = 1:size(curr_Features_test,1);
%     trainIdx_SUB{kk} = 1:size(curr_Features_train,1);
%     % [TrainAccuracy_SUB(kk), TestAccuracy_SUB(kk), Model_SUB{kk}] = classify_SVM_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),curr_YY_final','RBF',0,trainIdx{kk},testIdx{kk});
%     [TrainAccuracy_SUB(kk), TestAccuracy_SUB(kk), Model_SUB{kk}] = classify_SVMweighted_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),curr_YY_final','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%     
%     
%     % Run null classification (after randomly permuting the labels of the testing trials):
%     Y_null_train = curr_YY_final(trainIdx{kk}); Y_null = curr_YY_final;
%     Y_null(trainIdx{kk}) = Y_null_train(randperm(length(Y_null_train)));
%     % [TrainAccuracy_null_SUB(kk), TestAccuracy_null_SUB(kk), Model_null_SUB{kk}] = classify_SVM_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
%     [TrainAccuracy_null_SUB(kk), TestAccuracy_null_SUB(kk), Model_null_SUB{kk}] = classify_SVMweighted_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),Y_null','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%     
% end
% 
% save([curr_dir filesep Featurefiles_basename '_ClassificationResults_SUB'],'TrainAccuracy_SUB','TestAccuracy_SUB','Model_SUB','TrainAccuracy_null_SUB','TestAccuracy_null','Model_null','testIdx','trainIdx','current_test_block','current_train_blocks');
% 
% % Run Regression:
% curr_YY_final_continuous = curr_YY_final_continuous(find(trial_select_bin),:);
% for kk = 1:num_CV_folds % Make this par
%     [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
%     for tt = 1:size(curr_YY_final_continuous,2)
%         [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,curr_YY_final_continuous(:,tt),'RBF',0,trainIdx{kk},testIdx{kk});
%         % [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVMweighted_libsvm(curr_Features,YY_final_continuous(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%         % [Reg_TrainAccuracy(m,kk,tt), Reg_TestAccuracy(m,kk,tt), Reg_Model{kk,tt}] = classify_RSVM_libsvm(Features,YY_final_continuous(find(trial_select_bin),tt),'RBF',0,trainIdx{kk},testIdx{kk});
%         
%         % Y_null_train = curr_YY_final_continuous(trainIdx{kk},:); Y_null = curr_YY_final_continuous;
%         % Y_null(trainIdx{kk},:) = Y_null_train(randperm(size(Y_null_train,1)),:);
%         % [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
%         % [Reg_TrainAccuracy_null_SUB(kk,tt), Reg_TestAccuracy_null_SUB(kk,tt), Reg_Model_null_SUB{kk,tt}] = classify_RSVMweighted_libsvm(curr_Features,Y_null(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%         % [Reg_TrainAccuracy_null_SUB(kk,tt), Reg_TestAccuracy_null_SUB(kk,tt), Reg_Model_null_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,Y_null(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%         
%     end
% end
% 
%% Old code:

%         EEG = pop_biosig(curr_file);

% Properly read the events:
%         [~,header] = ReadEDF(curr_file);
%         for kk = 1:length(header.annotation.event) EEG.event(kk).type = header.annotation.event{kk}; end
%         task_conditions = unique(header.annotation.event);

%         % Select the trials that correspond to the classes selected:
%         trial_data_num = trial_data_num(nclassesIdx); Features = Features(nclassesIdx,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add individual trial markers (101, 201, etc..)
% Find the onset and duration of each trial in terms of the MRI volumes
%                         EEG_vhdr{m} = add_MRItrial_markers_EEGfMRI(EEG_vhdr{m}, curr_tD_file, scan_parameters ,m);
%                         EEG_mat{m} = add_MRItrial_markers_EEGfMRI(EEG_mat{m}, curr_tD_file, scan_parameters ,m);
%

%                         curr_class_markers_MRIonset = cell(1,length(curr_tD_file.class_MARKERS_BLOCKIDX_vect{m}));
%                         curr_class_markers_MRIduration = cell(1,length(curr_tD_file.class_MARKERS_BLOCKIDX_vect{m}));
%                         for idx_i = 1:length(curr_tD_file.class_MARKERS_BLOCKIDX_vect{m})
%                             x = curr_tD_file.class_MARKERS_BLOCKIDX_vect{m}{idx_i};
%                             for idx_j = 1:size(x,2)
%                                 curr_class_markers_MRIonset{idx_i}{idx_j} = find(diff(curr_tD_file.MRI_start_BLOCKIDX_vect{m} < x(1,idx_j))) + 1;
%                                 curr_class_markers_MRIduration{idx_i}{idx_j} = find(diff(curr_tD_file.MRI_end_BLOCKIDX_vect{m} > x(2,idx_j))) + 1;
%                             end
%
%                             % Replace empty entries with zero in curr_class_markers_MRI_onset
%                             temp_idx = find(cell2mat(cellfun(@(x)isempty(x),curr_class_markers_MRIonset{idx_i},'UniformOutput',0)));
%                             if ~isempty(temp_idx)
%                                 for mm = 1:length(temp_idx) curr_temp_idx = temp_idx(mm); curr_class_markers_MRIonset{idx_i}{curr_temp_idx} = 0; end
%                             end
%                             curr_class_markers_MRIonset{idx_i} = cell2mat(curr_class_markers_MRIonset{idx_i});
%
%                             % Replace empty entries with 1 in curr_class_markers_MRI_duration
%                             temp_idx = find(cell2mat(cellfun(@(x)isempty(x),curr_class_markers_MRIduration{idx_i},'UniformOutput',0)));
%                             if ~isempty(temp_idx)
%                                 for mm = 1:length(temp_idx) curr_temp_idx = temp_idx(mm); curr_class_markers_MRIduration{idx_i}{curr_temp_idx} = 1; end
%                             end
%                             curr_class_markers_MRIduration{idx_i} = cell2mat(curr_class_markers_MRIduration{idx_i})-curr_class_markers_MRIonset{idx_i};
%                         end
%
%                         % Add to EEG_vhdr:
%                         EEG_vhdr_SLICE_latency = find(cellfun(@(x) strcmp(x,scan_parameters.slice_marker),{EEG_vhdr{m}.event(:).type})); EEG_vhdr_SLICE_latency = cell2mat({EEG_vhdr{m}.event(EEG_vhdr_SLICE_latency).latency});
%                         EEG_vhdr_VOLUME_latency = EEG_vhdr_SLICE_latency(1:scan_parameters.slicespervolume:length(EEG_vhdr_SLICE_latency));
%                         for k = 1:length(curr_class_markers_MRIonset)
%                             curr_MRI_vol = curr_class_markers_MRIonset{k}; curr_MRI_withinbounds = (curr_MRI_vol>0);
%                             curr_MRI_vol = curr_MRI_vol(curr_MRI_withinbounds); % Remove the markers that are before the onset of the MRI scanning
%                             curr_latency = EEG_vhdr_VOLUME_latency(curr_MRI_vol); % curr_latency = curr_latency(curr_MRI_vol>0);
%                             curr_duration = curr_class_markers_MRIduration{k}(curr_MRI_withinbounds).*srate; % curr_duration = curr_duration(curr_MRI_vol>0);
%                             curr_type = mat2cell(curr_tD_file.class_MARKERS_vect{m}{k}(curr_MRI_withinbounds),1,ones(1,length(curr_duration))); % curr_type = curr_type(curr_MRI_vol>0);
%                             curr_type = cellfun(@(x){['MRI_' num2str(x)]},curr_type,'un',0);
%
%                             % First EEG_vhdr:
%                             EEG_vhdr{m} = add_events_from_latency_EEGfMRI(EEG_vhdr{m},curr_type, curr_latency,curr_duration);
%
%
%
% %                             for kk = 1:length(curr_class_markers_MRIonset{k})
% %                                 curr_MRI_vol = curr_class_markers_MRIonset{k}(kk);
% %                                 if curr_MRI_vol > 0
% %                                     n_events = length(EEG_vhdr{m}.event);
% %                                     EEG_vhdr{m}.event(n_events+1).type = curr_tD_file.class_MARKERS_vect{m}{k}(kk);
% %                                     EEG_vhdr{m}.event(n_events+1).latency = EEG_vhdr_VOLUME_latency(curr_MRI_vol);
% %                                     EEG_vhdr{m}.event(n_events+1).duration = curr_class_markers_MRIduration{k}(kk)*srate;
% %                                     EEG_vhdr{m}.event(n_events+1).urevent = n_events+1;
% %                                 end
% %                             end
%                         end
%                         % EEG_vhdr{m} = eeg_checkset(EEG_vhdr{m},'eventconsistency'); % Check for consistency and reorder the events chronologically
%
%                         % Add to EEG_mat:
%                         EEG_mat_SLICE_latency = find(cellfun(@(x) strcmp(x,scan_parameters.slice_marker),{EEG_mat{m}.event(:).type})); EEG_mat_SLICE_latency = cell2mat({EEG_mat{m}.event(EEG_mat_SLICE_latency).latency});
%                         EEG_mat_VOLUME_latency = EEG_mat_SLICE_latency(1:scan_parameters.slicespervolume:length(EEG_mat_SLICE_latency));
%                         for k = 1:length(curr_class_markers_MRIonset)
%                             curr_MRI_vol = curr_class_markers_MRIonset{k}; curr_MRI_withinbounds = (curr_MRI_vol>0);
%                             curr_MRI_vol = curr_MRI_vol(curr_MRI_withinbounds); % Remove the markers that are before the onset of the MRI scanning
%                             curr_latency = EEG_mat_VOLUME_latency(curr_MRI_vol); %curr_latency = curr_latency(curr_MRI_vol>0);
%                             curr_duration = curr_class_markers_MRIduration{k}(curr_MRI_withinbounds).*srate; % curr_duration = curr_duration(curr_MRI_vol>0);
%                             curr_type = mat2cell(curr_tD_file.class_MARKERS_vect{m}{k}(curr_MRI_withinbounds),1,ones(1,length(curr_duration))); % curr_type = curr_type(curr_MRI_vol>0);
%                             curr_type = cellfun(@(x){['MRI_' num2str(x)]},curr_type,'un',0);
%
%                             % Next EEG_mat:
%                             EEG_mat{m} = add_events_from_latency_EEGfMRI(EEG_mat{m},curr_type, curr_latency,curr_duration);
%
%                             % Add to EEG_mat:
%                             %for k = 1:length(curr_class_markers_MRIonset)
%                             %                             for kk = 1:length(curr_class_markers_MRIonset{k})
%                             %                                 curr_MRI_vol = curr_class_markers_MRIonset{k}(kk);
%                             %                                 if curr_MRI_vol > 0
%                             %                                     n_events = length(EEG_mat{m}.event);
%                             %                                     EEG_mat{m}.event(n_events+1).type = curr_tD_file.class_MARKERS_vect{m}{k}(kk);
%                             %                                     EEG_mat{m}.event(n_events+1).latency = curr_tD_file.MRI_start_BLOCKIDX_vect{m}(curr_MRI_vol);
%                             %                                     EEG_mat{m}.event(n_events+1).duration = curr_class_markers_MRIduration{k}(kk)*srate;
%                             %                                     EEG_mat{m}.event(n_events+1).urevent = n_events+1;
%                             %                                 end
%                             %                             end
%                         end
%                         % EEG_mat{m} = eeg_checkset(EEG_mat{m},'eventconsistency'); % Check for consistency and reorder the events chronologically
%
% DO THE EXACT SAME THING FOR ACTUAL INDICES AND NOT ALIGNED TO MRI VOLUMES

