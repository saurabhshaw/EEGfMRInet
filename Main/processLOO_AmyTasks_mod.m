
% file_ID = 'P1145_Pre';
% file_date = '20190521';
study_name = 'AmyTasks';
modality = 'EEGfMRI';

% Study specific parameters:
study_conditions = {'Relax','Sport','Navigation','Song','Subtraction','Finger tapping','Running'};
replace_files = 0;
scan_parameters = [];
scan_parameters.TR = 3.2; % MRI Repetition Time (in seconds)
scan_parameters.anat_num_images = 160;
scan_parameters.tfunc_num_images = 8560;
scan_parameters.slicespervolume = 40;
scan_parameters.slice_marker = 'R128';
scan_parameters.ECG_channel = 32;
scan_parameters.srate = 5000;
scan_parameters.low_srate = 500;

% Control Parameters:
runs_to_include = {'task'}; % From the description of the dataset can be 'task', 'rest' or 'combined'
overwrite_files = 0;
srate = 5000;
nclasses = [2 3]; % all = 1; left = 2; right = 3;
max_features = 1000;%keep this CPU-handle-able
testTrainSplit = 0.75; %classifier - trained on 25%
num_CV_folds_per_pt = 2; %classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'
final_featureSelection = 0; % Whether to run final feature selection after mRMR iterate - NOTE - this caused poor classification results
final_featuresToUse = 'individual'; % Can be 'preselected' or 'individualized'
final_featuresToUse_file = 'preselectedWindow_Model.mat'; % File to be used in case 'preselected'
run_matfile = 1;
save_finalData = 1; % Save the final Test/Training data for all participants in one master file

% Setup CONN_cfg:
CONN_cfg = [];
CONN_cfg.CONN_analysis_name = 'ROI'; % The name of the CONN first-level analysis
% CONN_cfg.CONN_analysis_name = 'V2V_02'; % The name of the CONN first-level analysis
% CONN_cfg.CONN_project_name = 'conn_composite_task_fMRI'; % The name of the CONN project
CONN_cfg.CONN_project_name = 'AmyTasks_EEGfMRI'; % The name of the CONN project
CONN_cfg.CONN_data_type = 'ICA'; % The source of the CONN data - can be ICA or ROI 
CONN_cfg.net_to_analyze = {'CEN', 'DMN', 'SN'}; % List all networks to analyze
CONN_cfg.use_All_cond = 1; % If this is 1, use the timecourses from condition 'All'
CONN_cfg.p_norm = 0; % Lp norm to use to normalize the timeseries data:  For data normalization - set to 0 to skip it, 1 = L1 norm and 2 = L2 norm
CONN_cfg.conditions_to_include = [1 2]; % The condition indices to sum up in the norm
CONN_cfg.window_step = 3.2; % in seconds - Used for Feature windows:
CONN_cfg.window_length = 9.6; % in seconds - Used for Feature windows:
CONN_cfg.threshold = 0.3; % Threshold for making the time course binary
CONN_cfg.class_types = 'networks'; % Can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze
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
window_step = 2; % in seconds
window_length = 10; % in seconds

% Set Paths:
base_path_main = fileparts(mfilename('fullpath')); cd(base_path_main); cd ..;
% base_path = pwd;
% base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
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
addpath(genpath([toolboxes_path filesep 'dimredtoolbox'])) % Adding Dimredtoolbox for TSNE
% addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

% Setup Data paths:
[base_path_rc, base_path_rd] = setPaths();
base_path_data = base_path_rd;
output_base_path_data = '/project/rrg-beckers/shaws5/Research_data'; %GRAHAM_OUTPUT_PATH
offline_preprocess_cfg.temp_file = [output_base_path_data filesep 'temp_deploy_files'];

distcomp.feature( 'LocalUseMpiexec', false );
mkdir('/tmp/jobstorage'); %GRAHAM_JOBSTORAGE_LOCATION

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
model_file = [base_path filesep 'Models' filesep 'SVM_Models' filesep final_featuresToUse_file];

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

% for ii = 2:length(sub_dir) % GRAHAM_PARFOR-1
%     for jj = 1:length(runs_to_include)
%         %%
%         try
%             fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
%             skip_analysis = 0;
%             dataset_to_use = [sub_dir(ii).name];
%             dataset_name = [sub_dir_mod(ii).PID];
%             curr_dir = [base_path_data filesep dataset_to_use]; %include single participant data-set
%             
%             %% Read in the file:
%             switch runs_to_include{jj}
%                 case 'task'
%                     task_DIR = dir([curr_dir filesep 'Task_block_*']);
%                     
%                 case 'rest'
%                     task_DIR = [1];
%             end
%             
%             %% Create output directory if not already made:
%             curr_dir = [output_base_path_data filesep dataset_to_use];
%             if isempty(dir(curr_dir))
%                 mkdir(curr_dir)
%             end
%             
%             %% Run analysis:
%             Features_SUB = cell(1,length(task_DIR));
%             YY_final_SUB = cell(1,length(task_DIR));
%             YY_final_continuous_SUB = cell(1,length(task_DIR));
%             
%             for m = 1:length(task_DIR) % GRAHAM_PARFOR-2
% 
%                 % Check if this task block has been processed by CONN:
%                 if ~isnan(EEGfMRI_corrIDX(ii,m))
%                     
%                     % Obtain the label vector:
%                     curr_CONN_IDX = EEGfMRI_corrIDX(ii,m);
%                     YY_final = cell2mat(CONN_data.fMRI_labels_selected_window_avg_thresh{ii-1}{curr_CONN_IDX});  % NOTE:only because this is excluding the first subject
%                     YY_final_continuous = (CONN_data.fMRI_labels_selected_window_avg{ii-1}{curr_CONN_IDX}); YY_final_continuous = cat(1,YY_final_continuous{:}); % NOTE:only because this is excluding the first subject
%                     YY_final_continuous_thresh = double(YY_final_continuous >= CONN_cfg.threshold);
%                     
%                     % Select relevant features:
%                     % nclassesIdx = randperm(length(YY_final));
%                     % [Features,Feature_labels_mRMR,Feature_mRMR_order] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features);
%                     % save([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults'],'Features','Feature_labels_mRMR','Feature_mRMR_order');
%                     task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
%                     Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
%                     Featurefiles_basename = ['Rev_' curr_dataset_name];
%                     
%                     % Pick the features to be used for the
%                     % classification:
%                     switch final_featuresToUse
%                         case 'individual'
%                             if isempty(dir([task_dir filesep 'EEG_Features' filesep '*_mRMRiterateResults.mat']))
%                                 curate_features_mRMR_compiled(Featurefiles_basename, Featurefiles_directory, YY_final, max_features, task_dir, base_path)
%                             end
%                             mRMRiterateResults = load([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults.mat']);
%                             Features = mRMRiterateResults.final_dataset_mRMR;
%                             
%                         case 'preselected'
%                             curr_model = load(model_file);
%                             tic; [Features] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile); toc;
%                     end
%                     
%                     if final_featureSelection
%                         if isempty(dir([task_dir filesep 'EEG_Features' filesep '*_FeatureSelectionResults.mat']))
%                             [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features,YY_final',max_features);
%                             save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR');
%                         else
%                             load([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults.mat']);
%                         end
%                         % [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features,YY_final',max_features);
%                         % save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR');
%                     else
%                         Features_ranked_mRMR = 1:size(Features,2);
%                         Features_scores_mRMR = nan(1,size(Features,2));
%                     end
%                     
%                     % Fix class imbalance and select Features and index:
%                     % [trial_select_bin] = fix_classImbalance(YY_final,'balance',0);
%                     [trial_select_bin,class_weights] = fix_classImbalance(YY_final,'balance',0);
%                     Features = Features(find(trial_select_bin),:); YY_final = YY_final(find(trial_select_bin)); YY_final_continuous_thresh = YY_final_continuous_thresh(find(trial_select_bin),:);
%                     
%                     % Accumulate Features for between task classification:
%                     Features_SUB{m} = Features;
%                     YY_final_SUB{m} = YY_final;
%                     YY_final_continuous_SUB{m} = YY_final_continuous;
%                     
%                     save([Featurefiles_directory filesep Featurefiles_basename '_FeaturesGEN'],'Features_SUB','YY_final_SUB','YY_final_continuous_SUB');
% 
%                     % load([Featurefiles_directory filesep Featurefiles_basename '_FeaturesGEN.mat']);
%                     
%                     fprintf(['\n ***************************** Finished Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ' num2str(m) ' ***************************** \n']);
%                 else
%                     fprintf(['\n ***************************** Skipping Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ' num2str(m) ' ***************************** \n']);
%                     
%                 end
%             end
%             
%             %% Accumulate the features for all subjects:
%             % curr_Features = cat(1,Features_SUB{:}); curr_YY_final = cat(2,YY_final_SUB{:}); curr_YY_final_continuous = cat(1,YY_final_continuous_SUB{:});
%             Features_ALLSUB{ii} = cat(1,Features_SUB{:}); YY_final_ALLSUB{ii} = cat(2,YY_final_SUB{:}); YY_final_continuous_ALLSUB{ii} = cat(1,YY_final_continuous_SUB{:});
% 
%             save([curr_dir filesep runs_to_include{jj} '_' dataset_name '_FeaturesALLSUBGEN'],'Features_ALLSUB','YY_final_ALLSUB','YY_final_continuous_ALLSUB');
% 
%         catch e
%             warning(['Problem with Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ]);
%             fprintf(1,'\n The error identifier was:\n%s',e.identifier);
%             fprintf(1,'\n There was an error! The message was:\n%s',e.message);
%             
%             % Write error file:
%             failure_string = getReport(e);
%             fileID = fopen(['ErrorLog_P' sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID '_Run_' runs_to_include{jj} '.txt'],'w');
%             fprintf(fileID,failure_string); fclose(fileID);
%         end
%     end
% end % GRAHAM_PARFOR_END

%% Create Group Matfile Array:
% sub_dir; sub_dir_mod; output_base_path_data; EEGfMRI_corrIDX; CONN_data
jj = 1;

Featurefiles_directory_ALL = []; Featurefiles_basename_ALL = []; curr_Featurefiles_basename_ALL = [];
YY_final_ALL = []; YY_final_continuous_ALL = []; subIDX_ALL = []; sessIDX_ALL = [];
for ii = 1:length(sub_dir) % GRAHAM_PARFOR-1
    skip_analysis = 0;
    dataset_to_use = [sub_dir(ii).name];
    dataset_name = [sub_dir_mod(ii).PID];
    curr_dir = [output_base_path_data filesep dataset_to_use];
    
    temp_idx = ~isnan(EEGfMRI_corrIDX(ii,:)); % Check if this task block has been processed by CONN:

    for m = find(temp_idx) % GRAHAM_PARFOR-2
        
        % fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
        
        % missing_curate_features_mRMR = 0;
        
        % Obtain the label vector:
        % Its already in CONN_data
        curr_CONN_IDX = EEGfMRI_corrIDX(ii,m);
        YY_final = cell2mat(CONN_data.fMRI_labels_selected_window_avg_thresh{ii-1}{curr_CONN_IDX});  % NOTE:only because this is excluding the first subject
        YY_final_continuous = (CONN_data.fMRI_labels_selected_window_avg{ii-1}{curr_CONN_IDX}); YY_final_continuous = cat(1,YY_final_continuous{:}); % NOTE:only because this is excluding the first subject
        YY_final_continuous_thresh = double(YY_final_continuous >= CONN_cfg.threshold);
        
        % Obtain features:
        % nclassesIdx = randperm(length(YY_final));
        % [Features,Feature_labels_mRMR,Feature_mRMR_order] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features);
        % save([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults'],'Features','Feature_labels_mRMR','Feature_mRMR_order');
        task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
        Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
        Featurefiles_basename = ['Rev_' curr_dataset_name];
        
        curr_Featurefiles_basename = strsplit(Featurefiles_basename,'_CLASS');
        curr_Featurefiles_basename = curr_Featurefiles_basename{1};
        
        Featurefiles_directory_ALL{ii,m} = Featurefiles_directory;
        Featurefiles_basename_ALL{ii,m} = Featurefiles_basename;
        curr_Featurefiles_basename_ALL{ii,m} = curr_Featurefiles_basename;
        YY_final_ALL{ii,m} = YY_final; YY_final_continuous_ALL{ii,m} = YY_final_continuous;
        subIDX_ALL(ii,m) = ii; sessIDX_ALL(ii,m) = m;
        % load([Featurefiles_directory filesep curr_Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
        
    end
end

subIDX_ALL = subIDX_ALL(:); skipIDX = subIDX_ALL ~= 0;
% subIDX_ALL(subIDX_ALL == 0) = NaN; sessIDX_ALL(sessIDX_ALL == 0) = NaN;
Featurefiles_directory_ALL = Featurefiles_directory_ALL(:); Featurefiles_directory_ALL = Featurefiles_directory_ALL(skipIDX);
Featurefiles_basename_ALL = Featurefiles_basename_ALL(:); Featurefiles_basename_ALL = Featurefiles_basename_ALL(skipIDX);
curr_Featurefiles_basename_ALL = curr_Featurefiles_basename_ALL(:); curr_Featurefiles_basename_ALL = curr_Featurefiles_basename_ALL(skipIDX);
YY_final_ALL = YY_final_ALL(:); YY_final_ALL = YY_final_ALL(skipIDX);
YY_final_continuous_ALL = YY_final_continuous_ALL(:); YY_final_continuous_ALL = YY_final_continuous_ALL(skipIDX);
subIDX_ALL = subIDX_ALL(skipIDX);
sessIDX_ALL = sessIDX_ALL(:); sessIDX_ALL = sessIDX_ALL(skipIDX);

YY_final_All_subIDX = cellfun(@(x,y) repmat(y,[1 length(x)]),YY_final_ALL,mat2cell(subIDX_ALL,ones(size(subIDX_ALL))),'un',0);

Featurefiles_curated_dir = dir([Featurefiles_directory_ALL{1} filesep curr_Featurefiles_basename_ALL{1} '_AllEpochs_*.mat']);
% Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

%% Run Classification between participants, between task blocks:
% Create output directory if not already made:
features_to_include = [2];

curr_dir = [output_base_path_data filesep [sub_dir(end).name]];
if isempty(dir(curr_dir))
    mkdir(curr_dir)
end


curr_labels_mRMR = cat(2,YY_final_ALL{:}); % Get current labels for mRMR
curr_labels_mRMR_subIDX = cat(2,YY_final_All_subIDX{:}); % Get current labels for mRMR
curr_Features = []; curr_freqFeatures = [];
for i = sort(features_to_include) 
    % Load freq_dataset_mRMR:
    curr_freqFeatures_struct = load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'mRMRiterate_' currFeatures_curated{i} '_' CONN_cfg.class_types],'freq_dataset_mRMR');
    curr_freqFeatures_mat = cat(2,curr_freqFeatures_struct.freq_dataset_mRMR{:});
    % Collect these same features from the new dataset across all windows
    % and do kmeans to find similar subjects to this:
    
    
    % Use the "Coarse" classification to give the labels and run the final
    % stage of feature-wise feature selection just on the subset of the
    % subjects that are similar to this subject to find the feature set
    % that optimally separates the classes for this new test dataset
    
    curr_Features_struct = load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_' currFeatures_curated{i} '_mRMRiterateGroupResults_' CONN_cfg.class_types]);
    curr_Feature_vect = nan(length(curr_labels_mRMR),size(curr_Features_struct.final_dataset_mRMR,2));
    curr_freqFeature_vect = nan(length(curr_labels_mRMR),size(curr_freqFeatures_mat,2));

    curr_IDX = squeeze(curr_Features_struct.curr_dataset_mRMR_IDX);
    
    curr_Feature_vect(curr_IDX,:) = curr_Features_struct.final_dataset_mRMR;
    curr_freqFeature_vect(curr_IDX,:) = curr_freqFeatures_mat;

    curr_Features = cat(2,curr_Features,curr_Feature_vect);
    curr_freqFeatures = cat(2,curr_freqFeatures,curr_freqFeature_vect);

end
curr_YY_final = curr_labels_mRMR;
curr_YY_final_continuous = cat(2,YY_final_continuous_ALL{:});

% Load relevant Feature data:
% if ~exist('Features_ALLSUB','var')
%     load([curr_dir filesep runs_to_include{end} '_' sub_dir_mod(end).PID '_FeaturesALLSUBGEN'],'Features_ALLSUB','YY_final_ALLSUB','YY_final_continuous_ALLSUB');
% end

% Run Feature Selection:
% Features_SUB = Features_ALLSUB; YY_final_SUB = YY_final_ALLSUB; YY_final_continuous_SUB = YY_final_continuous_ALLSUB;
% curr_Features = cat(1,Features_SUB{:}); curr_YY_final = cat(2,YY_final_SUB{:}); curr_YY_final_continuous = cat(1,YY_final_continuous_SUB{:});

% Featurefiles_basename = [runs_to_include{end} '_ResultsLOO'];

%% Run Final Feature Selection and compute class weights:
num_top_feat = 2500;
if final_featureSelection
    curr_Features_isnan = sum(isnan(curr_Features),2) > 0;
    if isempty(dir([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_FeatureSelectionGroupResults_' CONN_cfg.class_types]))
        [Features_ranked_mRMR_SUB, Features_scores_mRMR_SUB] = mRMR_custom(curr_Features(~curr_Features_isnan,:),curr_YY_final(~curr_Features_isnan),max_features*3);
        save([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_FeatureSelectionGroupResults_' CONN_cfg.class_types],'Features_ranked_mRMR_SUB','Features_scores_mRMR_SUB');
    else
        load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_FeatureSelectionGroupResults_' CONN_cfg.class_types]);
    end
    Features_ranked_mRMR_SUB = Features_ranked_mRMR_SUB(1:num_top_feat);
else
    Features_ranked_mRMR_SUB = 1:size(curr_Features,2);
    Features_scores_mRMR_SUB = nan(1,size(curr_Features,2));
end

% Fix class imbalance and select Features and index:
[trial_select_bin,class_weights] = fix_classImbalance(curr_YY_final,'balance',0);
curr_Features = curr_Features(find(trial_select_bin),:); curr_YY_final = curr_YY_final(find(trial_select_bin)); curr_subIDX = curr_labels_mRMR_subIDX(find(trial_select_bin));

%% Visualize the clusters in the data using t-SNE:
X = curr_Features;
X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));
curr_Features_isnan = sum(isnan(X),2) > 0;
Ydata = visualize_SVM_SVs_tSNE(X(~curr_Features_isnan,:), curr_YY_final(~curr_Features_isnan),[],3);



%% Find clusters in the data:
num_clusters = 4;
stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);
[idx,C,sumd,D] = kmeans(curr_Features,num_clusters,'Options',options,'MaxIter',10000,...
    'Display','final','Replicates',10);

%% Run Classification:
Results_outputDir = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'Classification_Results'];
if isempty(dir(Results_outputDir))
    mkdir(Results_outputDir)
end
% Model_SUB = cell(length(sub_dir),num_CV_folds_per_pt); Model_null_SUB = cell(length(sub_dir),num_CV_folds_per_pt);
% TrainAccuracy_SUB = zeros(length(sub_dir),num_CV_folds_per_pt); TestAccuracy_SUB = zeros(length(sub_dir),num_CV_folds_per_pt);
% TrainAccuracy_null_SUB = zeros(length(sub_dir),num_CV_folds_per_pt); TestAccuracy_null_SUB = zeros(length(sub_dir),num_CV_folds_per_pt);
Model_SUB = cell(1,length(sub_dir)); Model_null_SUB = cell(1,length(sub_dir));
TrainAccuracy_SUB = zeros(1,length(sub_dir)); TestAccuracy_SUB = zeros(1,length(sub_dir));
TrainAccuracy_null_SUB = zeros(1,length(sub_dir)); TestAccuracy_null_SUB = zeros(1,length(sub_dir));
for current_test_block = 2:length(sub_dir)
    % Run regular classification:
    % Identify the block to keep as the testing data:
    % current_test_block{kk} = randi(length(sub_dir));
    % while isempty(Features_SUB{current_test_block{kk}}) current_test_block{kk} = randi(length(sub_dir)); end
    % while isempty(intersect(curr_subIDX,current_test_block{kk})) current_test_block{kk} = randi(length(sub_dir)); end
    current_train_blocks = setdiff([1:length(sub_dir)],current_test_block);
    
    % For testing only:
    % current_train_blocks{kk} = current_train_blocks{kk}(unique(randi(length(current_train_blocks{kk}),[1 8])));
    
    % Create dataset accordingly:
    select_testIDX = current_test_block == curr_subIDX;
    select_trainIDX = zeros(size(curr_subIDX)); for i = 1:length(current_train_blocks) curr_bool = current_train_blocks(i) == curr_subIDX; select_trainIDX = select_trainIDX | curr_bool; end
    % select_trainIDX = arrayfun(@(x) x == curr_subIDX,current_train_blocks{kk},'un',0);
    curr_Features_test = curr_Features(select_testIDX,:); curr_YY_final_test = curr_YY_final(select_testIDX);
    curr_Features_train = curr_Features(select_trainIDX,:); curr_YY_final_train = curr_YY_final(select_trainIDX);
    % curr_Features_test = cat(1,Features_SUB{current_test_block{kk}}); curr_YY_final_test = cat(2,YY_final_SUB{current_test_block{kk}});
    % curr_Features_train = cat(1,Features_SUB{current_train_blocks{kk}}); curr_YY_final_train = cat(2,YY_final_SUB{current_train_blocks{kk}});
    
    curr_Features_classify = [curr_Features_test; curr_Features_train]; curr_YY_final_classify = [curr_YY_final_test curr_YY_final_train];
    test_train_split_IDX = [ones(1,size(curr_Features_test,1)) zeros(1,size(curr_Features_train,1))]; % Ones for test and zeros for train
    curr_Features_classify_isnan = sum(isnan(curr_Features_classify),2) > 0; curr_Features_classify = curr_Features_classify(~curr_Features_classify_isnan,:);
    curr_YY_final_classify = curr_YY_final_classify(~curr_Features_classify_isnan);
    test_train_split_IDX = test_train_split_IDX(~curr_Features_classify_isnan);
    
    testIdx_SUB = find(test_train_split_IDX); trainIdx_SUB = find(~test_train_split_IDX);

    
    
    
    %parfor kk = 1:num_CV_folds_per_pt % make this par
    % testIdx_SUB{kk} = 1:size(curr_Features_test,1); trainIdx_SUB{kk} = 1:size(curr_Features_train,1);
    
    % [TrainAccuracy_SUB(kk), TestAccuracy_SUB(kk), Model_SUB{kk}] = classify_SVM_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),curr_YY_final','RBF',0,trainIdx{kk},testIdx{kk});
    [TrainAccuracy_SUB(current_test_block), TestAccuracy_SUB(current_test_block), Model_SUB{current_test_block}] = classify_SVMweighted_libsvm(curr_Features_classify(:,Features_ranked_mRMR_SUB),curr_YY_final_classify','RBF',0,class_weights,trainIdx_SUB,testIdx_SUB);
    
    % Run null classification (after randomly permuting the labels of the testing trials):
    Y_null_train = curr_YY_final_classify(trainIdx_SUB); Y_null = curr_YY_final_classify;
    Y_null(trainIdx_SUB) = Y_null_train(randperm(length(Y_null_train)));
    % [TrainAccuracy_null_SUB(kk), TestAccuracy_null_SUB(kk), Model_null_SUB{kk}] = classify_SVM_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
    [TrainAccuracy_null_SUB(current_test_block), TestAccuracy_null_SUB(current_test_block), Model_null_SUB{current_test_block}] = classify_SVMweighted_libsvm(curr_Features_classify(:,Features_ranked_mRMR_SUB),Y_null','RBF',0,class_weights,trainIdx_SUB,testIdx_SUB);
    %end
    
    final_saveName = [Results_outputDir filesep 'FinalResults_' study_name 'LOO' '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types 'NEW' '_Feat' arrayfun(@(x) num2str(x),features_to_include) '_CVrun' num2str(current_test_block)];
    save(final_saveName,'TrainAccuracy_SUB','TestAccuracy_SUB','Model_SUB','TrainAccuracy_null_SUB','TestAccuracy_null_SUB','Model_null_SUB','testIdx_SUB','trainIdx_SUB','current_test_block','current_train_blocks','trial_select_bin','class_weights','-v7.3');
    
    fprintf(['\n ***************************** Finished Subject ' sub_dir_mod(current_test_block).PID ' ***************************** \n']);
end


% Save final data:
final_saveName = [Results_outputDir filesep 'FinalResults_' study_name 'LOO' '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types 'NEW' '_Feat' arrayfun(@(x) num2str(x),features_to_include)];
% save([curr_dir filesep Featurefiles_basename '_ClassificationResults_LOO'],'TrainAccuracy_SUB','TestAccuracy_SUB','Model_SUB','TrainAccuracy_null_SUB','TestAccuracy_null_SUB','Model_null_SUB','testIdx_SUB','trainIdx_SUB','current_test_block','current_train_blocks');
% save(final_saveName,'TrainAccuracy_SUB','TestAccuracy_SUB','Model_SUB','TrainAccuracy_null_SUB','TestAccuracy_null_SUB','Model_null_SUB','testIdx_SUB','trainIdx_SUB','current_test_block','current_train_blocks','trial_select_bin','class_weights','-v7.3');
save(final_saveName,'-v7.3');

%% Run Regression:
% create_currentTrainTest = 0;
% if ~isempty(dir([curr_dir filesep Featurefiles_basename '_ClassificationResults_LOO.mat']))
%     load([curr_dir filesep Featurefiles_basename '_ClassificationResults_LOO.mat']);
% else
%     create_currentTrainTest = 1;
% end
% 
% Reg_Model_SUB = cell(size(curr_YY_final_continuous,2),num_CV_folds);
% % curr_YY_final_continuous = curr_YY_final_continuous(find(trial_select_bin),:);
% for kk = 1:num_CV_folds % Make this par
%     
%     % Identify the block to keep as the testing data:
%     if create_currentTrainTest
%         Reg_current_test_block{kk} = randi(length(sub_dir));
%         while isempty(Features_SUB{Reg_current_test_block{kk}}) Reg_current_test_block{kk} = randi(length(sub_dir)); end
%         Reg_current_train_blocks{kk} = setdiff([1:length(sub_dir)],Reg_current_test_block{kk});
%         
%         % For testing only:
%         Reg_current_train_blocks{kk} = Reg_current_train_blocks{kk}(unique(randi(length(Reg_current_train_blocks{kk}),[1 3])));
%     end
%     
%     % Create dataset accordingly:
%     curr_Features_test = cat(1,Features_SUB{Reg_current_test_block{kk}}); curr_YY_final_continuous_test = cat(2,YY_final_continuous_SUB{Reg_current_test_block{kk}});
%     curr_Features_train = cat(1,Features_SUB{Reg_current_train_blocks{kk}}); curr_YY_final_continuous_train = cat(2,YY_final_continuous_SUB{Reg_current_train_blocks{kk}});
%     
%     curr_Features = [curr_Features_test; curr_Features_train]; curr_YY_continuous_final = [curr_YY_final_continuous_test curr_YY_final_continuous_train];
%     Reg_testIdx_SUB{kk} = 1:size(curr_Features_test,1); Reg_trainIdx_SUB{kk} = 1:size(curr_Features_train,1);
%     
%     %[testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
%     for tt = 1:size(curr_YY_final_continuous,2)
%         [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,curr_YY_final_continuous(:,tt),'RBF',0,Reg_trainIdx_SUB{kk},Reg_testIdx_SUB{kk});
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
% % Save final data:
% final_saveName = [base_path_main filesep 'FinalResults_' study_name 'LOO' '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types];
% if ~isempty(dir(final_saveName))
%     save(final_saveName,'-append','Reg_TrainAccuracy_SUB','Reg_TestAccuracy_SUB','Reg_Model_SUB','Reg_testIdx_SUB','Reg_trainIdx_SUB','Reg_current_test_block','Reg_current_train_blocks');
% else
%     % save([curr_dir filesep Featurefiles_basename '_RegressionResults_LOO'],'Reg_TrainAccuracy_SUB','Reg_TestAccuracy_SUB','Reg_Model_SUB','testIdx_SUB','trainIdx_SUB','current_test_block','current_train_blocks');   
%     save(final_saveName,'Reg_TrainAccuracy_SUB','Reg_TestAccuracy_SUB','Reg_Model_SUB','Reg_testIdx_SUB','Reg_trainIdx_SUB','Reg_current_test_block','Reg_current_train_blocks','trial_select_bin','class_weights');
% end


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

