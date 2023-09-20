
% file_ID = 'P1145_Pre';
% file_date = '20190521';
study_name = 'HomewoodNFB';
modality = 'EEG';

% Control Parameters:
runs_to_include = {'rest'}; % Conditions - From the description of the dataset
seconds_to_cut = 20; srate = 500;
nclasses = [2 3]; % all = 1; left = 2; right = 3;
max_features = 1000; %keep this CPU-handle-able
testTrainSplit = 0.75; %classifier - trained on 25%
num_CV_folds = 20; %classifier - increased folds = more accurate but more computer-intensive??
curate_features_individually = true;
feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI','BPow'};
% feature_names = {'BPow'};
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
offline_preprocess_cfg.remove_electrodes = 0;
offline_preprocess_cfg.manualICA_check = 0;

% Setup Feature windows:
window_step = 5; % in seconds
window_length = 10; % in seconds

% Set Paths:
% base_path_main = fileparts(mfilename('fullpath')); cd(base_path_main); cd ..; % Run this for graham_parfor 
base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
base_path = pwd;

% Include path:
toolboxes_path = [base_path filesep 'Toolboxes']; base_path_main = [base_path filesep 'Main'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB']; happe_directory_path = [toolboxes_path filesep 'Happe']; %HAPPE
addpath(genpath(base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));
% addpath(genpath([toolboxes_path filesep 'libsvm-3.23' filesep 'windows'])); % Adding LibSVM
% addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
% addpath(genpath([toolboxes_path filesep 'Fieldtrip'])); rmpath(genpath([toolboxes_path filesep 'Fieldtrip' filesep 'compat'])); % Adding Fieldtrip
% load([toolboxes_path filesep 'MatlabColors.mat']);     

[base_path_rc, base_path_rd] = setPaths();
base_path_data = base_path_rd;
output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
temp_filelocation = [output_base_path_data filesep 'temp_deploy_files'];

distcomp.feature( 'LocalUseMpiexec', false );
% mkdir('/tmp/jobstorage'); %GRAHAM_JOBSTORAGE_LOCATION

%% Process the participants' data:
% dataset_to_use = [file_date '_' study_name '_' modality '-' file_ID];
% dataset_name = file_ID;
chanlocs_file = [base_path filesep 'Cap_files' filesep 'Standard-10-20-Cap16_FrontMod.ced'];
elec_transform_file = [base_path filesep 'Cap_files' filesep 'GTecNautilus2BrainVision.xlsx'];

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
% sub_dir = [1];
%% For Loop:
for ii = 1:length(sub_dir) % GRAHAM_PARFOR
    for jj = 1:length(runs_to_include)        
        try
            fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
            skip_analysis = 0;
            dataset_to_use = [sub_dir(ii).name];
            dataset_name = [sub_dir_mod(ii).PID '_' sub_dir_mod(ii).SID];
            curr_dir = [base_path_data filesep dataset_to_use]; %include single participant data-set
            
            %% Read in the file:
            switch runs_to_include{jj}
                case 'task'
                    curr_file = [curr_dir filesep 'composite_task_' dataset_name '_full_dataset.mat']; skip_analysis = isempty(dir(curr_file));
                    if ~skip_analysis
                        task_EEG = load(curr_file, 'EEG','class_MARKERS');
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
                        task_EEG = load([curr_dir filesep 'composite_task_' dataset_name '_full_dataset.mat'], 'EEG','class_MARKERS');
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
                if strcmp(runs_to_include{jj},'rest') EEG = pop_select(EEG,'time',[seconds_to_cut EEG.xmax-seconds_to_cut]); end
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
                cap2brainvision_elec_table = readtable(elec_transform_file);
                cap2brainvision_elec = cap2brainvision_elec_table{:,2};
                if iscell(cap2brainvision_elec) cap2brainvision_elec = cellfun(@str2num,cap2brainvision_elec); end
                if sum(isnan(cap2brainvision_elec)) % Find the index of the subset of electrodes in the file
                    cap2brainvision_elec_subset = find(~isnan(cap2brainvision_elec));
                    cap2brainvision_elec = cap2brainvision_elec(~isnan(cap2brainvision_elec)); 
                    save([curr_dir filesep 'Electrode_subset_IDX.mat'],'cap2brainvision_elec_subset');
                end
                if length(size(EEG.data)) == 2 EEG.data = EEG.data(cap2brainvision_elec,:);
                else EEG.data = EEG.data(cap2brainvision_elec,:,:); end
                
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
                
                %% Curate features:
                fprintf(['\n ***************************** Curating Computed Features ***************************** \n']);
                Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
                Featurefiles_basename = ['Rev_' dataset_name];
                %[compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);
                [compute_feat] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);

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
