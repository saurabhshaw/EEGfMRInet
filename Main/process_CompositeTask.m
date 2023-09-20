
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
output_base_path_data = '/project/rrg-beckers/shaws5/Research_data'; %GRAHAM_OUTPUT_PATH
% output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
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
for ii = 2:length(sub_dir) % GRAHAM_PARFOR-1
    for jj = 1:length(runs_to_include)
        %%
        try
            fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
            skip_analysis = 0;
            dataset_to_use = [sub_dir(ii).name];
            dataset_name = [sub_dir_mod(ii).PID];
            curr_dir = [base_path_data filesep dataset_to_use]; %include single participant data-set
            %curr_filedir = dir([curr_dir filesep '*.set']);
            %curr_file = [curr_dir filesep dataset_name '.set'];
            
            %% Read in the file:
            switch runs_to_include{jj}
                case 'task'
                    task_dir = dir([curr_dir filesep 'Task_block_*']);
                    curr_tD_filename = [curr_dir filesep dataset_name '_trial_data.mat'];
                    if ~isempty(dir(curr_tD_filename)) curr_tD_file = load(curr_tD_filename); end
                    
                    EEG_vhdr = cell(size(curr_tD_file.block_onset_vect)); EEG_mat = cell(size(curr_tD_file.block_onset_vect)); saved_final_data = cell(size(curr_tD_file.block_onset_vect));

%                     for m = 1:length(task_dir) % GRAHAM_PARFOR-2
%                         fprintf(['\n ***************************** Processing Task Block ' num2str(m) ' ***************************** \n']);
% 
%                         %% Load data from the backup .vhdr datafile:
%                         curr_file = [curr_dir filesep 'Task_block_' num2str(m) filesep dataset_name '_full_dataset_' num2str(m) '.vhdr']; skip_analysis = isempty(dir(curr_file));
%                         if ~skip_analysis
%                             [curr_filepath, curr_filename] = fileparts(curr_file);
%                             EEG_vhdr{m} = cust_loadbv(curr_filepath,[curr_filename '.vhdr']);
%                             EEG_vhdr{m} = pop_chanedit(EEG_vhdr{m},'load',{chanlocs_file 'filetype' 'autodetect'});
%                             EEG_vhdr{m}.setname = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)]; EEG_vhdr{m} = eeg_checkset(EEG_vhdr{m});
%                             
%                             % Correct the number of slice markers if there are extra from Pre-scanning procedures:
%                             [EEG_vhdr{m}] = remove_extra_slicemarkers_EEGfMRI(EEG_vhdr{m},scan_parameters,scan_parameters.tfunc_num_images);                    
%                         end
%                         
%                         %% Load data from the saved .mat datafiles:
%                         curr_block_dir = [curr_dir filesep curr_tD_file.task_blocks_dir(m).name];
%                         final_data = curate_matfiles_EEGfMRI(curr_block_dir,study_prefix,scan_parameters); final_EEG = final_data.final_EEG;
%                         [EEG_mat{m},final_data] = curate_mat2set_EEGfMRI(final_data,chanlocs_file,[runs_to_include{jj} '_' dataset_name],scan_parameters,EEG_vhdr{m}.event);
%                         EEG_mat{m}.setname = [EEG_mat{m}.setname '_TaskBlock' num2str(m)];
%                         saved_final_data{m} = final_data;
%                         
%                         %% Add condition markers to both EEG streams:
%                         % Update the condition details based on the aligned slice markers:
%                         if isfield(final_data,'removed_start_latencies') curr_tD_file = EEGfMRI_update_aligned_volumes(curr_tD_file,final_data,scan_parameters,0,m); end
%                         
%                         % Based on MRIonset and MRIduration as defined for the CONN processing:
%                         EEG_vhdr{m} = add_MRIcondition_markers_EEGfMRI(EEG_vhdr{m}, curr_tD_file, scan_parameters ,m);
%                         EEG_mat{m} = add_MRIcondition_markers_EEGfMRI(EEG_mat{m}, curr_tD_file, scan_parameters ,m);
% 
%                         % Based on RDA blocks of BLOCKIDX (The "curr_onset" is with respect to the first slice marker - curr_tD_file.MRI_start_BLOCKIDX_vect{m}(1))
%                         RDA_blocksize = final_data.final_EEG_EVENT(1).RDAblocksize;
%                         EEG_vhdr{m} = add_RDAcondition_markers_EEGfMRI(EEG_vhdr{m}, curr_tD_file, scan_parameters, RDA_blocksize,m);
%                         EEG_mat{m} = add_RDAcondition_markers_EEGfMRI(EEG_mat{m}, curr_tD_file, scan_parameters, RDA_blocksize,m);
% 
%                         %% Add individual trial markers (101, 201, etc..)
%                         % Find the onset and duration of each trial in terms of the MRI volumes
%                         EEG_vhdr{m} = add_MRItrial_markers_EEGfMRI(EEG_vhdr{m}, curr_tD_file, scan_parameters ,m);
%                         EEG_mat{m} = add_MRItrial_markers_EEGfMRI(EEG_mat{m}, curr_tD_file, scan_parameters ,m);
%                           
%                         % DO THE EXACT SAME THING FOR ACTUAL INDICES AND
%                         % NOT ALIGNED TO MRI VOLUMES - for EEG processing!!
%                         
%                         %% Save to file:
%                         % Make folder if not already made:
%                         if ~isdir([curr_filepath filesep 'EEGfMRI_Raw']) mkdir([curr_filepath filesep 'EEGfMRI_Raw']); end
%                         output_dir = [curr_filepath filesep 'EEGfMRI_Raw'];
%                         
%                         %%%%%%%%%%%%%%%%% Uncomment Later
%                         %%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%
%                         %pop_saveset(EEG_vhdr{m}, 'filename',EEG_vhdr{m}.setname,'filepath',output_dir);
%                         %pop_saveset(EEG_mat{m}, 'filename',EEG_mat{m}.setname,'filepath',output_dir);
%                         
%                     end % GRAHAM_PARFOR_END
                case 'rest'
                    curr_file = [curr_dir filesep 'rsEEG' filesep dataset_name '_resting_dataset.vhdr']; skip_analysis = isempty(dir(curr_file));
                    if ~skip_analysis    
                        [curr_filepath, curr_filename] = fileparts(curr_file);
                        [EEG] = cust_loadbv(curr_filepath,[curr_filename '.vhdr']);
                        % EEG = pop_select(EEG,'time',[seconds_to_cut EEG.xmax-seconds_to_cut]);
                        EEG = pop_chanedit(EEG,'load',{chanlocs_file 'filetype' 'autodetect'});
                        EEG.setname = [runs_to_include{jj} '_' dataset_name]; EEG = eeg_checkset(EEG);
                        
                        % Correct the number of slice markers if there are extra from Pre-scanning procedures:
                        [EEG] = remove_extra_slicemarkers_EEGfMRI(EEG,scan_parameters,scan_parameters.rsfunc_num_images);
                        
                        % Save to file:
                        % Make folder if not already made:
                        if ~isdir([curr_filepath filesep 'EEGfMRI_Raw']) mkdir([curr_filepath filesep 'EEGfMRI_Raw']); end
                        output_dir = [curr_filepath filesep 'EEGfMRI_Raw'];                        
                        pop_saveset(EEG, 'filename',EEG.setname,'filepath',output_dir);
                    end                    
            end
                        
            %% Create output directory if not already made:
            curr_dir = [output_base_path_data filesep dataset_to_use];
            if isempty(dir(curr_dir))
                mkdir(curr_dir)
            end
            
            %% Preprocess datafile if not already preprocessed:
            if ~skip_analysis
                %% Preprocess datafile if not already preprocessed: 
                switch runs_to_include{jj}
                    case 'task'
%                         % First run EEG_vhdr:
%                         for m = 1:length(EEG_vhdr) % GRAHAM_PARFOR-2
%                             tic;
%                             fprintf(['\n ***************************** Starting Pre-Processing Task Block VHDR ' num2str(m) ' ***************************** \n']);
% 
%                             task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
%                             EEG_vhdr{m} = run_task_EEGfMRI_preprocess(EEG_vhdr{m},task_dir,scan_parameters,dataset_name,'VHDR',offline_preprocess_cfg,overwrite_files,base_path,m);
%                             toc
%                         end % GRAHAM_PARFOR_END
%                         
                        % Second run EEG_mat:
%                         for m = 1:length(EEG_mat)
%                             tic;
%                             fprintf(['\n ***************************** Starting Pre-Processing Task Block MAT ' num2str(m) ' ***************************** \n']);
% 
%                             task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
%                             EEG_mat{m} = run_task_EEGfMRI_preprocess(EEG_mat{m},task_dir,scan_parameters,dataset_name,'MAT',offline_preprocess_cfg,overwrite_files,base_path,m);
%                             toc
%                         end
                        
                    case 'rest'
                        tic;
                        fprintf(['\n ***************************** Starting Pre-Processing Rest ***************************** \n']);

                        rest_dir = [curr_dir filesep 'rsEEG'];
                        [EEG] = run_rest_EEGfMRI_preprocess(EEG,rest_dir,scan_parameters,dataset_name,offline_preprocess_cfg,overwrite_files,base_path);
                        toc
                end
                
                
                %% Compute EEG features:
%                 for m = 1:length(EEG_vhdr) % GRAHAM_PARFOR-2                    
%                     EEG = EEG_vhdr{m};
%                     
%                     % Splice the dataset into sliding windows, creating Epochs to compute features over:
%                     if length(size(EEG.data)) == 2
%                         slice_latencies = [EEG.event(find(strcmp(scan_parameters.slice_marker,{EEG.event.type}))).latency]; last_slice_latency = slice_latencies(length(slice_latencies));
%                         vol_latencies = slice_latencies(1:scan_parameters.slicespervolume:length(slice_latencies));
%                         
%                         TR_window_step = CONN_cfg.window_step/scan_parameters.TR; % The window_step in terms of the TR
%                         TR_window_length = CONN_cfg.window_length/scan_parameters.TR; % The window_length in terms of the TR
%                         
%                         % [start_idx, end_idx] = create_windows(size(EEG.data,2), window_step*EEG.srate, window_length*EEG.srate);
%                         [vol_start_idx, vol_end_idx] = create_windows(length(vol_latencies), TR_window_step, TR_window_length); % Compute the start and end indicies in terms of MR volumes
%                         start_idx = vol_latencies(vol_start_idx); % Convert the start and end indices in terms of the EEG latencies
%                         end_idx = min((start_idx + (CONN_cfg.window_length*EEG.srate)),size(EEG.data,2));
%                         temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
%                         EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});
%                     end
%                     
%                     % Define directory names and paths:
%                     curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
%                     task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
%                     save([task_dir filesep curr_dataset_name '_FeatureEpochDefinitions' ],'start_idx','end_idx');
%                     
%                     % Compute Features:
%                     currFeatures_dir = dir([task_dir filesep 'EEG_Features' filesep 'Rev_' curr_dataset_name '_Epoch*.mat']);
%                     currFeatures_finished = cellfun(@(x) strsplit(x,{'Epoch','.mat'}),{currFeatures_dir.name},'un',0); currFeatures_finished = cellfun(@(x) str2num(x{2}),currFeatures_finished);
%                     epochs_to_process = setdiff(1:size(EEG.data,3),currFeatures_finished);
%                     if ~isempty(epochs_to_process)
%                         %if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
%                         fprintf(['\n ***************************** Starting Feature Computation ***************************** \n']);
%                         tic; compute_features_compiled(EEG,task_dir,curr_dataset_name,feature_names,base_path); toc
%                     else
%                         fprintf(['\n ***************************** Features Computed for All Epochs ***************************** \n']);
%                     end
%                     
%                     % Curate features:
%                     fprintf(['\n ***************************** Curating Computed Features ***************************** \n']);
%                     Featurefiles_directory = [task_dir filesep 'EEG_Features'];
%                     Featurefiles_basename = ['Rev_' curr_dataset_name];
%                     % [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);
%                     [compute_feat] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);
%                                         
%                 end % GRAHAM_PARFOR_END

                %% Run feature selection and classification:
                redo_classification = 1; % If '1', do the classification again, otherwise load results from dataset saved
                run_regression = 0; % To train a regression model or not
                name_suffix = '';
                TrainAccuracy = zeros(length(EEG_vhdr),num_CV_folds); TestAccuracy = zeros(length(EEG_vhdr),num_CV_folds);
                TrainAccuracy_null = zeros(length(EEG_vhdr),num_CV_folds); TestAccuracy_null = zeros(length(EEG_vhdr),num_CV_folds);
                Feature_labels_mRMR_SUB = cell(1,length(EEG_vhdr)); Features_SUB = cell(1,length(EEG_vhdr)); YY_final_SUB = cell(1,length(EEG_vhdr));

                num_classes = size(CONN_data.fMRI_labels_selected_window_avg{ii-1}{1}{1},2);
                Reg_TrainAccuracy = zeros(length(EEG_vhdr),num_CV_folds,num_classes); Reg_TestAccuracy = zeros(length(EEG_vhdr),num_CV_folds,num_classes);
                Reg_TrainAccuracy_null = zeros(length(EEG_vhdr),num_CV_folds,num_classes); Reg_TestAccuracy_null = zeros(length(EEG_vhdr),num_CV_folds,num_classes);
                
                for m = 1:length(EEG_vhdr) % GRAHAM_PARFOR-2
                    
                    % Check if this task block has been processed by CONN:
                    if ~isnan(EEGfMRI_corrIDX(ii,m))
                        
                        missing_curate_features_mRMR = 0;
                        
                        % Obtain the label vector:
                        curr_CONN_IDX = EEGfMRI_corrIDX(ii,m);
                        YY_final = cell2mat(CONN_data.fMRI_labels_selected_window_avg_thresh{ii-1}{curr_CONN_IDX});  % NOTE:only because this is excluding the first subject
                        YY_final_continuous = (CONN_data.fMRI_labels_selected_window_avg{ii-1}{curr_CONN_IDX}); YY_final_continuous = cat(1,YY_final_continuous{:}); % NOTE:only because this is excluding the first subject
                        YY_final_continuous_thresh = double(YY_final_continuous >= CONN_cfg.threshold);

                        % Select relevant features:
                        % nclassesIdx = randperm(length(YY_final));
                        % [Features,Feature_labels_mRMR,Feature_mRMR_order] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features);
                        % save([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults'],'Features','Feature_labels_mRMR','Feature_mRMR_order');
                        task_dir = [curr_dir filesep 'Task_block_' num2str(m)];  
                        Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
                        Featurefiles_basename = ['Rev_' curr_dataset_name];
                        
%                         if ~isempty(dir([task_dir filesep 'EEG_Features' filesep Featurefiles_basename '_mRMRiterateResults.mat']))
%                             copyfile([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults.mat'],[Featurefiles_directory filesep Featurefiles_basename '_CLASS' CONN_cfg.class_types 'NEW_mRMRiterateResults.mat']);
%                             disp(['Writing ' Featurefiles_basename]);
%                         end
                        
                        if ~isempty(dir([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults' name_suffix '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types '_NEW.mat'])) && ~redo_classification
                            fprintf(['\n ***************************** Classification already done - Loading results for run ' num2str(m) ' ***************************** \n']);
                            temp = load([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults' name_suffix '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types '_NEW']);
                            
                            TestAccuracy(m,:) = temp.TestAccuracy(m,:);
                            TestAccuracy_null(m,:) = temp.TestAccuracy_null(m,:);
                            TrainAccuracy(m,:) = temp.TrainAccuracy(m,:);
                            TrainAccuracy_null(m,:) = temp.TrainAccuracy_null(m,:);
                            
                        else
                            
                            fprintf(['\n ***************************** Running classification for run ' num2str(m) ' ***************************** \n']);
                            
                            % Pick the features to be used for the
                            % classification:
                            switch final_featuresToUse
                                case 'individual'
                                    if isempty(dir([task_dir filesep 'EEG_Features' filesep Featurefiles_basename '_CLASS' CONN_cfg.class_types 'NEW_mRMRiterateResults.mat']))
                                        missing_curate_features_mRMR = 1; % REMOVE THIS LATER
                                        % curate_features_mRMR_compiled(Featurefiles_basename, Featurefiles_directory, YY_final, max_features, task_dir, base_path)
                                    end
                                    if ~missing_curate_features_mRMR % REMOVE THIS LATER
                                        mRMRiterateResults = load([Featurefiles_directory filesep Featurefiles_basename '_CLASS' CONN_cfg.class_types 'NEW_mRMRiterateResults.mat']);
                                        Features = mRMRiterateResults.final_dataset_mRMR;
                                    end
                                case 'preselected'
                                    curr_model = load(model_file);
                                    tic; [Features] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile); toc;
                                    % load([Featurefiles_directory filesep Featurefiles_basename '_FeaturesAllTrialsGEN.mat'])
                                    name_suffix = 'GEN';
                            end
                            
                            if ~missing_curate_features_mRMR % REMOVE THIS LATER
                                
                                if final_featureSelection
                                    if isempty(dir([task_dir filesep 'EEG_Features' filesep '*_FeatureSelectionResults.mat']))
                                        [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features,YY_final',max_features);
                                        save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR');
                                    else
                                        load([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults.mat']);
                                    end
                                    % [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features,YY_final',max_features);
                                    % save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults'],'Features','Features_ranked_mRMR','Features_scores_mRMR');
                                else
                                    Features_ranked_mRMR = 1:size(Features,2);
                                    Features_scores_mRMR = nan(1,size(Features,2));
                                end
                                
                                % Fix class imbalance and select Features and index:
                                % [trial_select_bin] = fix_classImbalance(YY_final,'balance',0);
                                [trial_select_bin,class_weights] = fix_classImbalance(YY_final,'balance',0);
                                Features = Features(find(trial_select_bin),:); YY_final = YY_final(find(trial_select_bin)); YY_final_continuous_thresh = YY_final_continuous_thresh(find(trial_select_bin),:);
                                
                                % Run Classification:
                                Model = cell(1,num_CV_folds); Model_null = cell(1,num_CV_folds);
                                parfor kk = 1:num_CV_folds % Make this par
                                    % Run regular classification:
                                    [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
                                    % [TrainAccuracy(m,kk), TestAccuracy(m,kk), Model{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),YY_final','RBF',0,trainIdx{kk},testIdx{kk});
                                    [TrainAccuracy(m,kk), TestAccuracy(m,kk), Model{kk}] = classify_SVMweighted_libsvm(Features,YY_final','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
                                    
                                    % Run null classification (after randomly permuting the labels of the testing trials):
                                    Y_null_train = YY_final(trainIdx{kk}); Y_null = YY_final;
                                    Y_null(trainIdx{kk}) = Y_null_train(randperm(length(Y_null_train)));
                                    % [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
                                    [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVMweighted_libsvm(Features,Y_null','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
                                end
                                
                                % CHANGED THIS - CHANGE THIS BACK!
                                % save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults' name_suffix '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types '_NEW'],'TrainAccuracy','TestAccuracy','Model','TrainAccuracy_null','TestAccuracy_null','Model_null','testIdx','trainIdx');
                                
                                % Accumulate Features for between task classification:
                                Features_SUB{m} = Features;
                                Feature_labels_mRMR_SUB{m} = Feature_labels_mRMR_SUB;
                                YY_final_SUB{m} = YY_final;
                                YY_final_continuous_SUB{m} = YY_final_continuous;
                                
                                % Run Regression:
                                if run_regression
                                    YY_final_continuous = YY_final_continuous(find(trial_select_bin),:);
                                    Reg_Model = cell(num_CV_folds,size(YY_final_continuous,2)); Reg_Model_null = cell(num_CV_folds,size(YY_final_continuous,2));
                                    for kk = 1:num_CV_folds % Make this par
                                        [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
                                        for tt = 1:size(YY_final_continuous,2)
                                            [Reg_TrainAccuracy(m,kk,tt), Reg_TestAccuracy(m,kk,tt), Reg_Model{kk,tt}] = classify_RSVM_matlab(Features,YY_final_continuous(:,tt),'RBF',0,trainIdx{kk},testIdx{kk});
                                        end
                                    end
                                    
                                    save([Featurefiles_directory filesep Featurefiles_basename '_RegressionResults' name_suffix],'Reg_TrainAccuracy','Reg_TestAccuracy','Reg_Model','testIdx','trainIdx');
                                    
                                end
                            end
                       end
                    end
                end
                
                % Load the classification results:
                % m = length(EEG_vhdr);
                % task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
                % Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
                % Featurefiles_basename = ['Rev_' curr_dataset_name];
                % temp = load([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults']);
               All_TestAccuracy{ii} = TestAccuracy; All_TestAccuracy_null{ii} = TestAccuracy_null;
               All_TrainAccuracy{ii} = TrainAccuracy; All_TrainAccuracy_null{ii} = TrainAccuracy_null;
                
                %% Run Classification within participant, between task blocks:
                
%                 if ~isempty(dir([curr_dir filesep Featurefiles_basename '_ClassificationResults' name_suffix '_SUB.mat']))
%                     load([curr_dir filesep Featurefiles_basename '_ClassificationResults' name_suffix '_SUB']);
%                 else
%                     % Run Feature Selection:
%                     curr_Features = cat(1,Features_SUB{:}); curr_YY_final = cat(2,YY_final_SUB{:}); curr_YY_final_continuous = cat(1,YY_final_continuous_SUB{:});
%                     
%                     Featurefiles_basename = ['Rev_' runs_to_include{jj} '_' dataset_name '_VHDR'];
%                     
%                     if final_featureSelection
%                         [Features_ranked_mRMR_SUB, Features_scores_mRMR_SUB] = mRMR(curr_Features,curr_YY_final',max_features*3);
%                         save([Featurefiles_directory filesep Featurefiles_basename '_FeatureSelectionResults_SUB'],'Features_SUB','Features_ranked_mRMR_SUB','Feature_labels_mRMR_SUB','Features_scores_mRMR_SUB');
%                     else
%                         Features_ranked_mRMR_SUB = 1:size(curr_Features,2);
%                         Features_scores_mRMR_SUB = nan(1,size(curr_Features,2));
%                     end
%                     
%                     % Fix class imbalance and select Features and index:
%                     [trial_select_bin,class_weights] = fix_classImbalance(curr_YY_final,'balance',0);
%                     curr_Features = curr_Features(find(trial_select_bin),:); curr_YY_final = curr_YY_final(find(trial_select_bin));
%                     
%                     % Run Classification:
%                     
%                     Model_SUB = cell(1,num_CV_folds); Model_null_SUB = cell(1,num_CV_folds);
%                     for kk = 1:num_CV_folds % make this par
%                         % Run regular classification:
%                         % Identify the block to keep as the testing data:
%                         current_test_block{kk} = randi(length(EEG_vhdr));
%                         current_train_blocks{kk} = setdiff([1:length(EEG_vhdr)],current_test_block{kk});
%                         
%                         % Create dataset accordingly:
%                         curr_Features_test = cat(1,Features_SUB{current_test_block{kk}}); curr_YY_final_test = cat(2,YY_final_SUB{current_test_block{kk}});
%                         curr_Features_train = cat(1,Features_SUB{current_train_blocks{kk}}); curr_YY_final_train = cat(2,YY_final_SUB{current_train_blocks{kk}});
%                         
%                         curr_Features = [curr_Features_test; curr_Features_train]; curr_YY_final = [curr_YY_final_test curr_YY_final_train];
%                         testIdx_SUB{kk} = 1:size(curr_Features_test,1);
%                         trainIdx_SUB{kk} = 1:size(curr_Features_train,1);
%                         % [TrainAccuracy_SUB(kk), TestAccuracy_SUB(kk), Model_SUB{kk}] = classify_SVM_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),curr_YY_final','RBF',0,trainIdx{kk},testIdx{kk});
%                         [TrainAccuracy_SUB(kk), TestAccuracy_SUB(kk), Model_SUB{kk}] = classify_SVMweighted_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),curr_YY_final','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%                         
%                         
%                         % Run null classification (after randomly permuting the labels of the testing trials):
%                         Y_null_train = curr_YY_final(trainIdx{kk}); Y_null = curr_YY_final;
%                         Y_null(trainIdx{kk}) = Y_null_train(randperm(length(Y_null_train)));
%                         % [TrainAccuracy_null_SUB(kk), TestAccuracy_null_SUB(kk), Model_null_SUB{kk}] = classify_SVM_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
%                         [TrainAccuracy_null_SUB(kk), TestAccuracy_null_SUB(kk), Model_null_SUB{kk}] = classify_SVMweighted_libsvm(curr_Features(:,Features_ranked_mRMR_SUB),Y_null','RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%                         
%                     end
%                     
%                     save([curr_dir filesep Featurefiles_basename '_ClassificationResults' name_suffix '_SUB' '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types],'TrainAccuracy_SUB','TestAccuracy_SUB','Model_SUB','TrainAccuracy_null_SUB','TestAccuracy_null_SUB','Model_null_SUB','testIdx_SUB','trainIdx_SUB','current_test_block','current_train_blocks');
%                     
%                 end
%                 
%                 % Run Regression: NEED TO FIX THIS - FROM PROCESS_LOO...
%                 if run_regression
%                     curr_YY_final_continuous = curr_YY_final_continuous(find(trial_select_bin),:);
%                     for kk = 1:num_CV_folds % Make this par
%                         [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
%                         for tt = 1:size(curr_YY_final_continuous,2)
%                             [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,curr_YY_final_continuous(:,tt),'RBF',0,trainIdx{kk},testIdx{kk});
%                             % [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVMweighted_libsvm(curr_Features,YY_final_continuous(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%                             % [Reg_TrainAccuracy(m,kk,tt), Reg_TestAccuracy(m,kk,tt), Reg_Model{kk,tt}] = classify_RSVM_libsvm(Features,YY_final_continuous(find(trial_select_bin),tt),'RBF',0,trainIdx{kk},testIdx{kk});
%                             
%                             % Y_null_train = curr_YY_final_continuous(trainIdx{kk},:); Y_null = curr_YY_final_continuous;
%                             % Y_null(trainIdx{kk},:) = Y_null_train(randperm(size(Y_null_train,1)),:);
%                             % [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
%                             % [Reg_TrainAccuracy_null_SUB(kk,tt), Reg_TestAccuracy_null_SUB(kk,tt), Reg_Model_null_SUB{kk,tt}] = classify_RSVMweighted_libsvm(curr_Features,Y_null(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%                             % [Reg_TrainAccuracy_null_SUB(kk,tt), Reg_TestAccuracy_null_SUB(kk,tt), Reg_Model_null_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,Y_null(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%                             
%                         end
%                     end
%                     
%                     save([curr_dir filesep Featurefiles_basename '_RegressionResults' name_suffix '_SUB'],'Reg_TrainAccuracy_SUB','Reg_TestAccuracy_SUB','Reg_Model_SUB','testIdx_SUB','trainIdx_SUB','current_test_block','current_train_blocks');
%                     
%                 end
%                
                TestAccuracy_SUB = []; TrainAccuracy_SUB = []; TestAccuracy_null_SUB = []; TrainAccuracy_null_SUB = [];
                
                % Load the classification results:
                All_TestAccuracySUB{ii} = TestAccuracy_SUB; All_TestAccuracy_nullSUB{ii} = TestAccuracy_null_SUB;
                All_TrainAccuracySUB{ii} = TrainAccuracy_SUB; All_TrainAccuracy_nullSUB{ii} = TrainAccuracy_null_SUB;
                
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

if save_finalData
   finalData_name = [base_path_main filesep 'FinalResults_' study_name '_FEAT' final_featuresToUse '_CLASS' CONN_cfg.class_types '_NEW']; 
   save(finalData_name,'All_TestAccuracySUB','All_TestAccuracy_nullSUB', 'All_TrainAccuracySUB', 'All_TrainAccuracy_nullSUB',...
                       'All_TestAccuracy', 'All_TestAccuracy_null', 'All_TrainAccuracy', 'All_TrainAccuracy_null');    
end

%% Final data processing:
plotResults = 1;
jj = 1;

% Get the final accuracy data:
All_TestAccuracy_mean = nan(size(EEGfMRI_corrIDX)); All_TestAccuracy_null_mean = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_mean = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_null_mean = nan(size(EEGfMRI_corrIDX));
All_TestAccuracy_std = nan(size(EEGfMRI_corrIDX)); All_TestAccuracy_null_std = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_std = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_null_std = nan(size(EEGfMRI_corrIDX));

All_TrainAccuracy_ttest = []; All_TestAccuracy_ttest = [];
for ii = 2:length(All_TestAccuracy)
    temp_idx = ~isnan(EEGfMRI_corrIDX(ii,:));
    All_TestAccuracy_mean(ii,temp_idx) = mean(All_TestAccuracy{ii}(temp_idx,:),2);
    All_TestAccuracy_null_mean(ii,temp_idx) = mean(All_TestAccuracy_null{ii}(temp_idx,:),2);
    All_TrainAccuracy_mean(ii,temp_idx) = mean(All_TrainAccuracy{ii}(temp_idx,:),2);
    All_TrainAccuracy_null_mean(ii,temp_idx) = mean(All_TrainAccuracy_null{ii}(temp_idx,:),2);
    
    All_TestAccuracy_std(ii,temp_idx) = std(All_TestAccuracy{ii}(temp_idx,:),[],2);
    All_TestAccuracy_null_std(ii,temp_idx) = std(All_TestAccuracy_null{ii}(temp_idx,:),[],2);
    All_TrainAccuracy_std(ii,temp_idx) = std(All_TrainAccuracy{ii}(temp_idx,:),[],2);
    All_TrainAccuracy_null_std(ii,temp_idx) = std(All_TrainAccuracy_null{ii}(temp_idx,:),[],2);
    
    % Test if the Accuracies of the Null model and actual model are different: 
    for tt = find(temp_idx)
        [All_TrainAccuracy_ttest{ii,tt}.h,All_TrainAccuracy_ttest{ii,tt}.p,All_TrainAccuracy_ttest{ii,tt}.ci,All_TrainAccuracy_ttest{ii,tt}.stats] = ttest2(All_TrainAccuracy{ii}(tt,:),All_TrainAccuracy_null{ii}(tt,:));
        [All_TestAccuracy_ttest{ii,tt}.h,All_TestAccuracy_ttest{ii,tt}.p,All_TestAccuracy_ttest{ii,tt}.ci,All_TestAccuracy_ttest{ii,tt}.stats] = ttest2(All_TestAccuracy{ii}(tt,:),All_TestAccuracy_null{ii}(tt,:));
        
        
        % Plot histograms of the top features:
        if plotResults
            dataset_to_use = [sub_dir(ii).name];
            dataset_name = [sub_dir_mod(ii).PID];
            curr_dir = [output_base_path_data filesep dataset_to_use];
            task_dir = [curr_dir filesep 'Task_block_' num2str(tt)];
            Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(tt)];
            Featurefiles_basename = ['Rev_' curr_dataset_name];
            
            mRMRiterateResults = load([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults.mat']);
            
            % NEED TO UPDATE THIS
            final_feature_labels_mRMR_vect = cellfun(@(x)strsplit(x,'_'),mRMRiterateResults.final_feature_labels_mRMR,'UniformOutput',0);
            features_vect = cell2mat(cellfun(@(x)str2num(x{1}),final_feature_labels_mRMR_vect,'UniformOutput',0));
            window_lengths_vect = cell2mat(cellfun(@(x)str2num(x{2}),final_feature_labels_mRMR_vect,'UniformOutput',0));
            window_step_vect = cell2mat(cellfun(@(x)str2num(x{3}),final_feature_labels_mRMR_vect,'UniformOutput',0));
            freq_bands_vect = cell2mat(cellfun(@(x)str2num(x{4}),final_feature_labels_mRMR_vect,'UniformOutput',0));
            chans1_vect = cell2mat(cellfun(@(x)str2num(x{length(x)-1}),final_feature_labels_mRMR_vect,'UniformOutput',0));
            chans2_vect = cell2mat(cellfun(@(x)str2num(x{length(x)}),final_feature_labels_mRMR_vect,'UniformOutput',0));
            
            subsess_output_scores(1) = subsess_output_scores(2);
            
            % Save histograms:
            % Chans:
            chans_vect = [chans1_vect chans2_vect];
            chans = unique(chans_vect); chans_hist = zeros(1,length(chans));
            for i = 1:length(chans)
                chans_hist(i) = sum(subsess_output_scores(chans1_vect==chans(i))) + sum(subsess_output_scores(chans2_vect==chans(i)));
            end
            bar(chans_hist);
            % hist([chans1_vect chans2_vect]);
            print('-djpeg','-r500',[fileNameTxt '_Chans_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
            % figure; topoplot2((chans_hist - mean(chans_hist))./std(chans_hist),1:51);
            % print('-djpeg','-r500',['Chans_topoplot_Sub' num2str(subject_number) '_Ses' num2str(session_number)]);
            
            
            % Freq Bands [full, delta, theta, alpha, beta, gamma]:
            freq_bands = unique(freq_bands_vect); freq_bands_hist = zeros(1,length(freq_bands));
            for i = 1:length(freq_bands)
                freq_bands_hist(i) = sum(subsess_output_scores(freq_bands_vect==freq_bands(i)));
            end
            bar(freq_bands_hist);
            % hist([freq_bands_vect]);
            print('-djpeg','-r500',[fileNameTxt '_Freq_Bands_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
            
            % Window lengths [100 200 400 1000]:
            window_lengths = unique(window_lengths_vect); window_lengths_hist = zeros(1,length(window_lengths));
            for i = 1:length(window_lengths)
                window_lengths_hist(i) = sum(subsess_output_scores(window_lengths_vect==window_lengths(i)));
            end
            bar(window_lengths_hist);
            % hist(window_lengths_vect);
            print('-djpeg','-r500',[fileNameTxt '_Window_Lengths_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
            
            %     % Window steps [50 100 200 500]:
            %     window_step = unique(window_step_vect); window_step_hist = zeros(1,length(window_step));
            %     for i = 1:length(window_step)
            %         window_step_hist(i) = sum(subsess_output_scores(window_step_vect==window_step(i)));
            %     end
            %     bar(window_step_hist);
            %     % hist([freq_bands_vect]);
            %     hist(window_step_vect);
            %     print('-djpeg','-r500',['Window_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]);
            %
            
            % Features: [1 7 8 9 10 11]
            features = feature_nums_to_do; features_hist = zeros(1,length(features));
            for i = 1:length(features)
                features_hist(i) = sum(subsess_output_scores(features_vect==features(i)));
            end
            bar(features_hist);
            % hist([freq_bands_vect]);
            % hist(features_vect);
            print('-djpeg','-r500',[fileNameTxt '_Features_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
        end
    end
end

if plotResults
    
    final_feature_labels_mRMR_vect = cellfun(@(x)strsplit(x,'_'),subsess_feature_labels_mRMR,'UniformOutput',0);
    features_vect = cell2mat(cellfun(@(x)str2num(x{1}),final_feature_labels_mRMR_vect,'UniformOutput',0));
    window_lengths_vect = cell2mat(cellfun(@(x)str2num(x{2}),final_feature_labels_mRMR_vect,'UniformOutput',0));
    window_step_vect = cell2mat(cellfun(@(x)str2num(x{3}),final_feature_labels_mRMR_vect,'UniformOutput',0));
    freq_bands_vect = cell2mat(cellfun(@(x)str2num(x{4}),final_feature_labels_mRMR_vect,'UniformOutput',0));
    chans1_vect = cell2mat(cellfun(@(x)str2num(x{length(x)-1}),final_feature_labels_mRMR_vect,'UniformOutput',0));
    chans2_vect = cell2mat(cellfun(@(x)str2num(x{length(x)}),final_feature_labels_mRMR_vect,'UniformOutput',0));
    
    subsess_output_scores(1) = subsess_output_scores(2);
    
    % Save histograms:
    % Chans:
    chans_vect = [chans1_vect chans2_vect];
    chans = unique(chans_vect); chans_hist = zeros(1,length(chans));
    for i = 1:length(chans)
        chans_hist(i) = sum(subsess_output_scores(chans1_vect==chans(i))) + sum(subsess_output_scores(chans2_vect==chans(i)));
    end
    bar(chans_hist);
    % hist([chans1_vect chans2_vect]);
    print('-djpeg','-r500',[fileNameTxt '_Chans_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
    % figure; topoplot2((chans_hist - mean(chans_hist))./std(chans_hist),1:51);
    % print('-djpeg','-r500',['Chans_topoplot_Sub' num2str(subject_number) '_Ses' num2str(session_number)]);
    
    
    % Freq Bands [full, delta, theta, alpha, beta, gamma]:
    freq_bands = unique(freq_bands_vect); freq_bands_hist = zeros(1,length(freq_bands));
    for i = 1:length(freq_bands)
        freq_bands_hist(i) = sum(subsess_output_scores(freq_bands_vect==freq_bands(i)));
    end
    bar(freq_bands_hist);
    % hist([freq_bands_vect]);
    print('-djpeg','-r500',[fileNameTxt '_Freq_Bands_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
    
    % Window lengths [100 200 400 1000]:
    window_lengths = unique(window_lengths_vect); window_lengths_hist = zeros(1,length(window_lengths));
    for i = 1:length(window_lengths)
        window_lengths_hist(i) = sum(subsess_output_scores(window_lengths_vect==window_lengths(i)));
    end
    bar(window_lengths_hist);
    % hist(window_lengths_vect);
    print('-djpeg','-r500',[fileNameTxt '_Window_Lengths_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
    
    %     % Window steps [50 100 200 500]:
    %     window_step = unique(window_step_vect); window_step_hist = zeros(1,length(window_step));
    %     for i = 1:length(window_step)
    %         window_step_hist(i) = sum(subsess_output_scores(window_step_vect==window_step(i)));
    %     end
    %     bar(window_step_hist);
    %     % hist([freq_bands_vect]);
    %     hist(window_step_vect);
    %     print('-djpeg','-r500',['Window_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]);
    %
    
    % Features: [1 7 8 9 10 11]
    features = feature_nums_to_do; features_hist = zeros(1,length(features));
    for i = 1:length(features)
        features_hist(i) = sum(subsess_output_scores(features_vect==features(i)));
    end
    bar(features_hist);
    % hist([freq_bands_vect]);
    % hist(features_vect);
    print('-djpeg','-r500',[fileNameTxt '_Features_hist_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
    
    switch model_type
        case 'SVM_libsvm'
            % Visualize Support Vectors:
            sv = Model{ii,num_runs}.SVs;
            visualize_SVM_SVs_tSNE(subsess_dataset_mRMR, curr_labels_mRMR, sv, 2)
            print('-djpeg','-r500',[fileNameTxt '_Visualize_SV_Sub' num2str(subject_number) '_Ses' num2str(session_number)]); close all;
    end
end

% fprintf(fileID, ['\n\nSub ' num2str(subject_number) ' Ses ' num2str(session_number) ' Accuracies with ' num2str(length(curr_labels_mRMR)) ' epochs and ' num2str(num_runs) ' testing periods']);
% fprintf(fileID, ['\nFeature\t= \t' num2str(feature_nums_to_do) '\n']);
% fprintf(fileID, ['Train Accuracy Average\t= \t' num2str(mean(TrainAccuracy(ii,:))*100) '   %% St.Dev \t= \t' num2str(std(TrainAccuracy(ii,:))*100) '  %% \n']);
% fprintf(fileID, ['Test Accuracy Average\t= \t' num2str(mean(TestAccuracy(ii,:))*100) '   %% St.Dev \t= \t' num2str(std(TestAccuracy(ii,:))*100) '  %% \n']);
% fprintf(fileID, ['Null Train Accuracy Average\t= \t' num2str(mean(TrainAccuracy_null(ii,:))*100) '   %% St.Dev \t= \t' num2str(std(TrainAccuracy_null(ii,:))*100) '  %% \n']);
% fprintf(fileID, ['Null Test Accuracy Average\t= \t' num2str(mean(TestAccuracy_null(ii,:))*100) '   %% St.Dev \t= \t' num2str(std(TestAccuracy_null(ii,:))*100) '  %% \n']);
% fprintf(fileID, ['Train Accuracy ttest: h\t= \t' num2str(TrainAccuracy_ttest{ii}.h) '   %% p \t= \t' num2str(TrainAccuracy_ttest{ii}.p) '  %% \n']);
% fprintf(fileID, ['Test Accuracy ttest: h\t= \t' num2str(TestAccuracy_ttest{ii}.h) '   %% p \t= \t' num2str(TestAccuracy_ttest{ii}.p) '  %% \n']);


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




% Run Regression:
% if run_regression
%     curr_YY_final_continuous = curr_YY_final_continuous(find(trial_select_bin),:);
%     for kk = 1:num_CV_folds % Make this par
%         [testIdx{kk}, trainIdx{kk}] = testTrainIdx_overlap(size(Features,1),CONN_cfg.window_length/scan_parameters.TR,1-testTrainSplit);
%         for tt = 1:size(curr_YY_final_continuous,2)
%             [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,curr_YY_final_continuous(:,tt),'RBF',0,trainIdx{kk},testIdx{kk});
%             % [Reg_TrainAccuracy_SUB(kk,tt), Reg_TestAccuracy_SUB(kk,tt), Reg_Model_SUB{kk,tt}] = classify_RSVMweighted_libsvm(curr_Features,YY_final_continuous(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%             % [Reg_TrainAccuracy(m,kk,tt), Reg_TestAccuracy(m,kk,tt), Reg_Model{kk,tt}] = classify_RSVM_libsvm(Features,YY_final_continuous(find(trial_select_bin),tt),'RBF',0,trainIdx{kk},testIdx{kk});
%             
%             % Y_null_train = curr_YY_final_continuous(trainIdx{kk},:); Y_null = curr_YY_final_continuous;
%             % Y_null(trainIdx{kk},:) = Y_null_train(randperm(size(Y_null_train,1)),:);
%             % [TrainAccuracy_null(m,kk), TestAccuracy_null(m,kk), Model_null{kk}] = classify_SVM_libsvm(Features(:,Features_ranked_mRMR),Y_null','RBF',0,trainIdx{kk},testIdx{kk});
%             % [Reg_TrainAccuracy_null_SUB(kk,tt), Reg_TestAccuracy_null_SUB(kk,tt), Reg_Model_null_SUB{kk,tt}] = classify_RSVMweighted_libsvm(curr_Features,Y_null(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%             % [Reg_TrainAccuracy_null_SUB(kk,tt), Reg_TestAccuracy_null_SUB(kk,tt), Reg_Model_null_SUB{kk,tt}] = classify_RSVM_matlab(curr_Features,Y_null(:,tt),'RBF',0,class_weights,trainIdx{kk},testIdx{kk});
%             
%         end
%     end
% end
