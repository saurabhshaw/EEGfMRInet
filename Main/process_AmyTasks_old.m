
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
num_CV_folds = 20; %classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'

% Setup CONN_cfg:
CONN_cfg = [];
CONN_cfg.CONN_analysis_name = 'V2V_02'; % The name of the CONN first-level analysis
CONN_cfg.CONN_project_name = 'conn_composite_task_fMRI'; % The name of the CONN project
CONN_cfg.CONN_data_type = 'ICA'; % The source of the CONN data - can be ICA or ROI 
CONN_cfg.net_to_analyze = {'CEN', 'DMN', 'SN'}; % List all networks to analyze
CONN_cfg.use_All_cond = 0; % If this is 1, use the timecourses from condition 'All'
CONN_cfg.p_norm = 1; % Lp norm to use to normalize the timeseries data
CONN_cfg.conditions_to_include = [1 2]; % The condition indices to sum up in the norm
CONN_cfg.window_step = 3.2; % in seconds - Used for Feature windows:
CONN_cfg.window_length = 9.6; % in seconds - Used for Feature windows:
CONN_cfg.threshold = 0.02; % Threshold for making the time course binary
CONN_cfg.class_types = 'subnetworks'; % Can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze
CONN_cfg.multilabel = 0; % Whether classification is multilabel or single label

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
window_step = 3.2; % in seconds
window_length = 9.6; % in seconds

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
addpath(genpath([toolboxes_path filesep 'libsvm-3.23' filesep 'windows'])); % Adding LibSVM
addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
addpath(genpath([toolboxes_path filesep 'Mricron'])) % Adding Mricron
% addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

[base_path_rc, base_path_rd] = setPaths();
base_path_data = base_path_rd;
output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
offline_preprocess_cfg.temp_file = [output_base_path_data filesep 'temp_deploy_files'];

distcomp.feature( 'LocalUseMpiexec', false );
% mkdir('/tmp/jobstorage'); %GRAHAM_JOBSTORAGE_LOCATION

%% Process the participants' data:
% dataset_to_use = [file_date '_' study_name '_' modality '-' file_ID];
% dataset_name = file_ID;
study_prefix = 'composite_task_';

chanlocs_file = [base_path filesep 'Cap_files' filesep 'BrainProductsMR64_NZ_LPA_RPA_fixed.sfp'];

% Create Subject Table/JSON file:
% [sub_dir,sub_dir_mod] = conn_addSubjects_EEGfMRI_PRT(study_name,modality,base_path_data, scan_parameters,study_conditions, replace_files);
[sub_dir,sub_dir_mod,sub_onset,sub_duration] = conn_addSubjects_EEGfMRI_PRT(study_name,modality,base_path_data, scan_parameters,study_conditions, replace_files);
% [sub_dir,sub_dir_mod] = conn_addSubjects_corrections_EEGfMRI(study_name,modality,base_path_data, scan_parameters,study_conditions,chanlocs_file,replace_files);
% [sub_dir,sub_dir_mod] = update_subject_list(study_name,modality,base_path_data,runs_to_include);

%% sub_dir = [1];
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
                    
                    EEG_vhdr = cell(size(curr_tD_file.block_onset_vect)); 
                    
                    for m = 1:length(task_dir) % GRAHAM_PARFOR-2
                        fprintf(['\n ***************************** Processing Task Block ' num2str(m) ' ***************************** \n']);

                        %% Load data from the backup .vhdr datafile:
                        curr_file = [curr_dir filesep 'Task_block_' num2str(m) filesep dataset_name(end-1:end) '_IN' num2str(m) '.vhdr']; skip_analysis = isempty(dir(curr_file));
                        if ~skip_analysis
                            [curr_filepath, curr_filename] = fileparts(curr_file);
                            EEG_vhdr{m} = cust_loadbv(curr_filepath,[curr_filename '.vhdr']);
                            EEG_vhdr{m} = pop_chanedit(EEG_vhdr{m},'load',{chanlocs_file 'filetype' 'autodetect'});
                            EEG_vhdr{m}.setname = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)]; EEG_vhdr{m} = eeg_checkset(EEG_vhdr{m});
                            
                            % Correct the number of slice markers if there are extra from Pre-scanning procedures:
                            % Changed the threshold to 10 from 20 for Composite task
                            [EEG_vhdr{m}] = remove_extra_slicemarkers_EEGfMRI(EEG_vhdr{m},scan_parameters,scan_parameters.tfunc_num_images);                    
                        end
                        
                        %% Add condition markers to both EEG streams:
                        % Based on MRIonset and MRIduration as defined for the CONN processing:
                        EEG_vhdr{m} = add_MRIcondition_markers_EEGfMRI(EEG_vhdr{m}, curr_tD_file, scan_parameters ,m);
                        
                        %% Save to file:
                        % Make folder if not already made:
                        if ~isdir([curr_filepath filesep 'EEGfMRI_Raw']) mkdir([curr_filepath filesep 'EEGfMRI_Raw']); end
                        output_dir = [curr_filepath filesep 'EEGfMRI_Raw'];
                        
                        %%%%%%%%%%%%%%%%% Uncomment Later
                        %%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%
                        %pop_saveset(EEG_vhdr{m}, 'filename',EEG_vhdr{m}.setname,'filepath',output_dir);
                        %pop_saveset(EEG_mat{m}, 'filename',EEG_mat{m}.setname,'filepath',output_dir);
                        
                    end % GRAHAM_PARFOR_END
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
                        % First run EEG_vhdr:
                        for m = 1:length(EEG_vhdr) % GRAHAM_PARFOR-2
                            tic;
                            fprintf(['\n ***************************** Starting Pre-Processing Task Block VHDR ' num2str(m) ' ***************************** \n']);

                            task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
                            EEG_vhdr{m} = run_task_EEGfMRI_preprocess(EEG_vhdr{m},task_dir,scan_parameters,dataset_name,'VHDR',offline_preprocess_cfg,overwrite_files,base_path,m);
                            toc
                        end % GRAHAM_PARFOR_END
                        
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
                
                for m = 1:length(EEG_vhdr) % GRAHAM_PARFOR-2                    
                    EEG = EEG_vhdr{m};
                    
                    % Splice the dataset into sliding windows, creating Epochs to compute features over:
                    if length(size(EEG.data)) == 2
                        slice_latencies = [EEG.event(find(strcmp(scan_parameters.slice_marker,{EEG.event.type}))).latency]; last_slice_latency = slice_latencies(length(slice_latencies));
                        vol_latencies = slice_latencies(1:scan_parameters.slicespervolume:length(slice_latencies));
                        
                        TR_window_step = round(CONN_cfg.window_step/scan_parameters.TR); % The window_step in terms of the TR
                        TR_window_length = round(CONN_cfg.window_length/scan_parameters.TR); % The window_length in terms of the TR
                        
                        % [start_idx, end_idx] = create_windows(size(EEG.data,2), window_step*EEG.srate, window_length*EEG.srate);
                        [vol_start_idx, vol_end_idx] = create_windows(length(vol_latencies), TR_window_step, TR_window_length); % Compute the start and end indicies in terms of MR volumes
                        start_idx = vol_latencies(vol_start_idx); % Convert the start and end indices in terms of the EEG latencies
                        end_idx = min((start_idx + (CONN_cfg.window_length*EEG.srate)),size(EEG.data,2));
                        temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
                        EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});
                    end
                    
                    % Define directory names and paths:
                    curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
                    task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
                    save([task_dir filesep curr_dataset_name '_FeatureEpochDefinitions' ],'start_idx','end_idx');
                    
                    % Compute Features:
                    currFeatures_dir = dir([task_dir filesep 'EEG_Features' filesep 'Rev_' curr_dataset_name '_Epoch*.mat']);
                    currFeatures_finished = cellfun(@(x) strsplit(x,{'Epoch','.mat'}),{currFeatures_dir.name},'un',0); currFeatures_finished = cellfun(@(x) str2num(x{2}),currFeatures_finished);
                    epochs_to_process = setdiff(1:size(EEG.data,3),currFeatures_finished);
                    if ~isempty(epochs_to_process)
                        %if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
                        fprintf(['***************************** Starting Feature Computation *****************************']);
                        tic; compute_features_compiled(EEG,task_dir,curr_dataset_name,feature_names,base_path); toc
                    end
                    
                    % Curate features:
                    fprintf(['***************************** Curating Computed Features *****************************']);
                    Featurefiles_directory = [task_dir filesep 'EEG_Features'];
                    Featurefiles_basename = ['Rev_' curr_dataset_name];
                    [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
                    
                end % GRAHAM_PARFOR_END
                
                %         if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
                %             fprintf(['***************************** Starting Feature Computation *****************************']);
                %             compute_features_attentionbci
                %         end
                %
                %         % Curate Features once computed:
                %         fprintf(['***************************** Curating Computed Features *****************************']);
                %
                %         Featurefiles_basename = ['Rev_Sub' num2str(ii) '_Ses' num2str(jj)];
                %         Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
                %         [compute_feat, Features] = curate_features_attentionbci(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
                %
                %
                %         %% Select Features and Classify the data:
                %         fprintf(['***************************** Starting Feature Selection and Classification *****************************']);
                %
                %         % Select the trials corresponding to the classes specified above:
                %         % trial_data = cellfun(@(x)
                %         % x{1},{EEG.epoch(:).eventtype},'UniformOutput',0); % Needed for Matlab versions before R2019b
                %         trial_data = cellfun(@(x) x,{EEG.epoch(:).eventtype},'UniformOutput',0); trial_data_unique = unique(trial_data); % Convert labels to numbers
                %         % Need to change this for R2019b = automatically take in the trial values, etc.
                %         [~,trial_data_num] = ismember(trial_data,trial_data_unique); % Convert labels to numbers
                %         nclassesIdx = []; for i = 1:length(nclasses) nclassesIdx = [nclassesIdx find(trial_data_num == nclasses(i))]; end % Pick the trials corresponding to the relevant classes
                %
                %         % Run Feature Selection:
                %         [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features(nclassesIdx,:),trial_data_num(nclassesIdx)',max_features);
                %
                %         % Classify:
                %         TrainAccuracy = zeros(1,num_CV_folds); TestAccuracy = zeros(1,num_CV_folds); Model = cell(1,num_CV_folds);
                %         parfor i = 1:num_CV_folds
                %             [TrainAccuracy(i), TestAccuracy(i), Model{i}] = classify_SVM_libsvm(Features(nclassesIdx,Features_ranked_mRMR),trial_data_num(nclassesIdx)','RBF',testTrainSplit);
                %         end
                %
                %         save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'TrainAccuracy','TestAccuracy','Model','Features_ranked_mRMR','Features_scores_mRMR','nclasses');
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

