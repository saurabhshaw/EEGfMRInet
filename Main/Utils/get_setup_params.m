function [general_param, scan_param, control_param,EEGfMRI_preprocess_param,EEG_preprocess_param, feature_param, CONN_param] = get_setup_params()

%% GENERAL PARAMS (ie filepaths)
% set file naming params
general_param.study_name = 'HCWMIVO';
general_param.modality = 'EEGfMRI';
data_subset_folder = 'dataset_2';
repo_filepath = 'C:\Users\DaniWorkstation\Documents_local\neuroscience_phd\research_code\EEGfMRInet';

% setting base paths
general_param.base_path = repo_filepath;
general_param.base_path_src = [repo_filepath filesep 'Main'];
toolboxes_path = [general_param.base_path filesep 'Toolboxes'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB'];

% add paths to wd 
addpath(genpath(general_param.base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));

% set research code/research data paths
[~,host_name] = system('hostname'); host_name = strtrim(host_name);
switch host_name        
    case 'DESKTOP-8S2HATP' % Dan home PC
        general_param.base_path_rd = 'D:\research_data';

    case 'MSI' % Dan laptop
        general_param.base_path_rd = 'D:\research_data';
end

% define runs and conditions
general_param.dict_runs_w_conditions = dictionary('task',{['MInjury','-','Neutral']},'rest',{['rsEEG_Post','-','rsEEG_Pre']}); %sep with '-' when adding new conditions
general_param.runs = keys(general_param.dict_runs_w_conditions);
runs_to_include = {};
for i = 1:length(general_param.runs)
    runs_to_include = [runs_to_include general_param.runs(i)];
end

% get base paths
general_param.base_path_data = [general_param.base_path_rd filesep general_param.study_name];
general_param.base_path_data = [general_param.base_path_data filesep data_subset_folder];
[general_param.sub_dir,general_param.sub_dir_mod] = update_subject_list(general_param.study_name,general_param.modality,general_param.base_path_data,runs_to_include);

% parfor configs -- distcomp, distributed computing
distcomp.feature( 'LocalUseMpiexec', false );

%% SCAN PARAMS
scan_param.TR = 3; % MRI Repetition Time (in seconds) i.e., time btwn volumes
scan_param.slicespervolume = 57;% check
scan_param.rsfunc_num_volumes = 160; % # of volumes per each rest condition run
scan_param.tfunc_num_volumes = 118; % # of volumes per each task condition run
scan_param.slice_marker = -1;
scan_param.ECG_channel = 67; % heart beat channel
scan_param.rsfunc_num_images = scan_param.rsfunc_num_volumes * scan_param.slicespervolume; % num of images per each rest condition run
scan_param.tfunc_num_images = scan_param.tfunc_num_volumes * scan_param.slicespervolume; % num of images per each task condition run
scan_param.low_srate = 500;

%% CONTROL PARAMS
control_param.overwrite_files = 0;

%% PREPROCESSING PARAMS - eegfmri
EEGfMRI_preprocess_param.use_fastr_gui = 0; % whether or not to use the fastr gui
EEGfMRI_preprocess_param.use_pas_gui = 0; % whether or not to use the fastr gui
EEGfMRI_preprocess_param.lpf = 70; % Low Pass Filter Cut-off
EEGfMRI_preprocess_param.L = 10; % Interpolation folds
EEGfMRI_preprocess_param.window = 30; % Averaging window
EEGfMRI_preprocess_param.Trigs = []; % Trigger Vector
EEGfMRI_preprocess_param.strig = 1; % 0-Slice Triggers or 1-Volume Triggers
EEGfMRI_preprocess_param.anc_chk = 1; % Run ANC
EEGfMRI_preprocess_param.tc_chk = 0; % Run Slice timing correction
EEGfMRI_preprocess_param.rel_pos = 0.03; % relative position of slice trig from beginning of slice acq
                             % 0 for exact start -> 1 for exact end
                             % default = 0.03;
EEGfMRI_preprocess_param.exclude_chan = [scan_param.ECG_channel]; % Channels not to perform OBS  on.
EEGfMRI_preprocess_param.num_PC = 'auto'; % Number of PCs to use in OBS.
EEGfMRI_preprocess_param.parallel = 'cpu'; % Can be cpu or gpu(under development)
EEGfMRI_preprocess_param.EEGLAB_preprocess_BPfilter = 0; % eeg data (btwn 0.1hz - 70hz) ==> NOTE: currently, this filtering is being done later - namely, in the general preprocessing
EEGfMRI_preprocess_param.EEGLAB_preprocess_Nfilter = 0; %%IN DEVELOPMENT%% slice selection freq (~20Hz), vibration noise (~26Hz) AC power line (~60Hz) ==> NOTE: currently, this filtering is being done during feature computation
EEGfMRI_preprocess_param.low_bp_filt = 0.1; EEGfMRI_preprocess_param.high_bp_filt = 70; EEGfMRI_preprocess_param.slice_selection_n_filt = 20; EEGfMRI_preprocess_param.vibrations_n_filt = 26; EEGfMRI_preprocess_param.ac_power_n_filt = 60;
EEGfMRI_preprocess_param.ICA_data_select = 1;% Whether to prune data or not
EEGfMRI_preprocess_param.ICA_data_select_range = 5; % Number of seconds before and after the last slice marker to keep for ICA analysis
EEGfMRI_preprocess_param.fastr_data_append_length = 1; % The length of time to append the EEG data before FASTR - to avoid index out of bounds errors
EEGfMRI_preprocess_param.fastr_data_append_threshold = 0.5; % The minimum length of time after the last slice marker which will trigger the appending of extra data.
EEGfMRI_preprocess_param.qrs_event_marker = 'qrs';

%% PREPROCESSING PARAMS - eeg
EEG_preprocess_param.preMRItruncation = 1; % Number of seconds to keep before the start of the MRI slice markers
EEG_preprocess_param.temp_file = [general_param.base_path_data filesep 'temp_deploy_files'];
EEG_preprocess_param.filter_lp = 0.1; % Was 1 Hz
EEG_preprocess_param.filter_hp = 50; % Was 40 Hz
EEG_preprocess_param.segment_data = 0; % 1 for epoched data, 0 for continuous data
EEG_preprocess_param.segment_markers = {}; % {} is all
EEG_preprocess_param.task_segment_start = -0.5; % start of the segments in relation to the marker
EEG_preprocess_param.task_segment_end = 5; % end of the segments in relation to the marker
EEG_preprocess_param.ChannelCriterion = 0.8; % To reject channels based on similarity to other channels
EEG_preprocess_param.run_second_ICA = 0;
EEG_preprocess_param.save_Workspace = 0;
EEG_preprocess_param.overwrite_files = 0;
EEG_preprocess_param.remove_electrodes = 1;
EEG_preprocess_param.manualICA_check = 0;

%% FEATURE PARAMS
feature_param.max_features = 1000; % Keep this CPU-handle-able
feature_param.testTrainSplit = 0.75; % Classifier - trained on 25%
feature_param.num_CV_folds = 20; % Classifier - increased folds = more accurate but more computer-intensive??
feature_param.curate_features_individually = true;
feature_param.feature_names = {'COH', 'PAC', 'dPLI', 'CFC_SI'}; %  {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'}; potentially remove dPLI?
feature_param.featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'
feature_param.final_featureSelection = 0; % Whether to run final feature selection after mRMR iterate - NOTE - this caused poor classification results
feature_param.final_featuresToUse = 'individual'; % Can be 'preselected' or 'individualized'
feature_param.final_featuresToUse_file = 'preselectedWindow_Model.mat'; % File to be used in case 'preselected'
feature_param.run_matfile = 1;
feature_param.save_finalData = 1; % Save the final Test/Training data for all participants in one master file

%% CONN PARAMS
CONN_param.CONN_analysis_name = 'ROI'; % The name of the CONN first-level analysis
% CONN_param.CONN_analysis_name = 'V2V_02'; % The name of the CONN first-level analysis
% CONN_param.CONN_project_name = 'conn_composite_task_fMRI'; % The name of the CONN project
CONN_param.CONN_project_name = 'HCWMIVO'; % The name of the CONN project
CONN_param.CONN_data_type = 'ICA'; % The source of the CONN data - can be ICA or ROI
CONN_param.net_to_analyze = {'CEN', 'DMN', 'SN'}; % List all networks to analyze
CONN_param.use_All_cond = 1; % If this is 1, use the timecourses from condition 'All'
CONN_param.p_norm = 0; % Lp norm to use to normalize the timeseries data:  For data normalization - set to 0 to skip it, 1 = L1 norm and 2 = L2 norm
CONN_param.conditions_to_include = [1 2]; % The condition indices to sum up in the norm
CONN_param.threshold = 0.3; % Threshold for making the time course binary
CONN_param.class_types = 'networks'; % Can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze
CONN_param.multilabel = 0; % Whether classification is multilabel or single label
CONN_param.ROIs_toUse = {'ICA_CEN','ICA_LCEN','ICA_anteriorSN','ICA_posteriorSN','ICA_ventralDMN','ICA_dorsalDMN'}; % Need this if using ROIs for labels rather than ICA
CONN_param.rescale = 1; % Rescale between 0 and 1 if selected


