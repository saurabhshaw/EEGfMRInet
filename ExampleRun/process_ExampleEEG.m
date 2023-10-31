
% Set participant specifics:
% base_path = 'C:\Users\danme\Documents_local\neuroscience_phd\research_code\EEGfMRInet'; % Base path of the package (your repository path)
% curr_condition = 'EyesClosed';

%% get req'd parameters
% addPaths
[general_param, scan_param, control_param,EEGfMRI_preprocess_param,EEG_preprocess_param, feature_param, CONN_param] = get_example_setup_params();

% PREP DATA FOR PRE-PROCESSING
% get participant data location & data
curr_dir = general_param.base_path_data;
curry_file_dir = dir([curr_dir filesep '*.cdt']);

%% read and save raw participant data, if exists
skip_analysis = isempty(curry_file_dir);
if ~skip_analysis
    
    curry_file = [curry_file_dir.folder filesep curry_file_dir.name];
    [curry_filepath, curry_filename] = fileparts(curry_file);
    [EEG] = loadcurry(curry_file);
    
    % check EEG field consistencies
    EEG.setname = [curr_run '_' general_param.participant_id '_' general_param.curr_condition]; EEG = eeg_checkset(EEG);
    
    % save to file
    % create folder if dne
    if ~isfolder([curry_filepath filesep 'EEGfMRI_Raw'])
        mkdir([curry_filepath filesep 'EEGfMRI_Raw']);
    end
    output_dir = [curry_filepath filesep 'EEGfMRI_Raw'];
    pop_saveset(EEG, 'filename',EEG.setname,'filepath',output_dir);
    
    % START PRE-PROCESSING
    fprintf(['\n ***************************** Starting Pre-Processing Task ***************************** \n']);
    
    if strcmp(curr_run,'task')
        num_volumes = scan_param.tfunc_num_volumes;
    elseif strcmp(curr_run,'rest')
        num_volumes = scan_param.rsfunc_num_volumes;
    end
    
    % check if slice marker injection is needed
    if sum(cellfun(@(x)x == scan_param.slice_marker,{EEG.event(:).type})) < num_volumes
        [EEG] = inject_missing_markers(EEG,EEG.srate,scan_param.slice_marker,num_volumes,scan_param.TR);
    end
    
    % sanity check slice marker injection success
    if sum(cellfun(@(x)x == scan_param.slice_marker,{EEG.event(:).type})) == num_volumes
        tic
        [EEG] = EEGfMRI_preprocess_full(EEG,curr_dir,scan_param,num_volumes,EEG_preprocess_param,EEGfMRI_preprocess_param,control_param.overwrite_files);
        toc
    end
    
    % BEGIN FEATURE COMPUTATION
    
    % define epochs
    slice_latencies = floor([EEG.event(find(strcmp(num2str(scan_param.slice_marker),{EEG.event.type}))).latency]);
    start_idx = min(slice_latencies);
    max_idx = max(slice_latencies);
    append_idx = start_idx;
    window_length = scan_param.TR*EEG.srate;
    window_step = scan_param.TR*EEG.srate;
    while (start_idx(end) < max_idx)
        start_idx = [start_idx, append_idx+window_step];
        append_idx = start_idx(end);
    end
    end_idx = floor(start_idx + window_length)-1;
    
    % avoid out of bounds errors + maintain max samples
    while end_idx(end) > size(EEG.data,2)
        end_idx = end_idx - 1;
    end
    
    % save epoch definitions
    save([curr_dir filesep EEG.setname '_FeatureEpochDefinitions' ],'start_idx','end_idx');
    
    % create epochs from definitions
    temp_data = arrayfun(@(x,y) EEG.data(:,x:y),start_idx,end_idx,'un',0); temp_time = arrayfun(@(x,y) EEG.times(1,x:y),start_idx,end_idx,'un',0);
    EEG.data = cat(3,temp_data{:}); EEG.times = cat(3,temp_time{:});
    
    % COMPUTE FEATURES
    currFeatures_dir = dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' EEG.setname '_Epoch*.mat']);
    currFeatures_finished = cellfun(@(x) strsplit(x,{'Epoch'}),{currFeatures_dir.name},'un',0);
    currFeatures_finished = cellfun(@(x) strsplit(x{2},{'.mat'}),currFeatures_finished,'un',0);
    currFeatures_finished = cellfun(@(x) str2num(x{1}),currFeatures_finished);
    epochs_to_process = setdiff(1:size(EEG.data,3),currFeatures_finished);
    if ~isempty(epochs_to_process)
        %if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
        fprintf(['\n ***************************** Starting Feature Computation ***************************** \n']);
        tic; compute_features_compiled(EEG,curr_dir,EEG.setname,feature_param.feature_names,general_param.base_path); toc
    else
        fprintf(['\n ***************************** Features Computed for All Epochs ***************************** \n']);
    end
    
    % CURATE FEATURES
    fprintf(['\n ***************************** Curating Computed Features ***************************** \n']);
    Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
    Featurefiles_basename = ['Rev_' EEG.setname];
    % [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);
    [compute_feat] = curate_features_deploy(feature_param.feature_names, feature_param.featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);
    
else
    fprintf(['\n ********** CDT FILE MISSING :: Processing Subject: ' general_param.sub_dir_mod(kk).PID ', Run: ' curr_run ', Condition: ' general_param.curr_condition ' ********** \n']);
end