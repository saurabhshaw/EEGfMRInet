

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
num_CV_folds_per_pt = 2; % Classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'
final_featureSelection = 0; % Whether to run final feature selection after mRMR iterate - NOTE - this caused poor classification results
final_featuresToUse = 'preselected'; % Can be 'preselected' or 'individualized'
final_featuresToUse_file = 'preselectedWindow_Model.mat'; % File to be used in case 'preselected'
run_matfile = 1;

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
EEGfMRI_corr = readtable([base_path_rd filesep 'EEGfMRI_ProcessingCorrespondence_' study_name '.xls'],'Format','auto'); % This is to make sure that the blocks of the EEG and fMRI correspond together
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
