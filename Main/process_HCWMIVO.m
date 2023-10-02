% get req'd parameters
[general_param, scan_param, control_param,EEGfMRI_preprocess_param,EEG_preprocess_param, feature_param, CONN_param] = get_setup_params();

for kk = 1:length(general_param.sub_dir)
    for ii = 1:length(general_param.runs)
        curr_run = general_param.runs{ii};
        conditions = cellfun(@(x) strsplit(x,'-'),general_param.dict_runs_w_conditions(curr_run),'UniformOutput',false);
        conditions = conditions{1,1};
        for jj = 1:length(conditions)
            curr_condition = conditions{1,jj};
            % try %dmer1

                % PREP DATA FOR PRE-PROCESSING
                fprintf(['\n ***************************** Processing Subject: ' general_param.sub_dir_mod(kk).PID ', Run: ' curr_run ', Condition: ' curr_condition ' ***************************** \n']);
                
                % get participant data location & data
                participant_dataset = general_param.sub_dir(kk).name;
                participant_id = general_param.sub_dir_mod(kk).PID;
                curr_dir = [general_param.base_path_data filesep participant_dataset];
                condition_dir = [curr_dir filesep 'EEG' filesep curr_condition];
                curry_file_dir = dir([condition_dir filesep '*.cdt']);
                
                % read and save raw participant data, if exists
                skip_analysis = isempty(curry_file_dir);
                if ~skip_analysis

                    curry_file = [curry_file_dir.folder filesep curry_file_dir.name];
                    [curry_filepath, curry_filename] = fileparts(curry_file);
                    [EEG] = loadcurry(curry_file);
  
                    % check EEG field consistencies
                    EEG.setname = [curr_run '_' participant_id '_' curr_condition]; EEG = eeg_checkset(EEG);

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
                    [EEG] = EEGfMRI_preprocess_full(EEG,condition_dir,scan_param,num_volumes,EEG_preprocess_param,EEGfMRI_preprocess_param,control_param.overwrite_files);
                    toc
                    end
                    
                    % begin feature computation

                    % define windows & save definitions
                    [EEG] = create_windows(EEG, scan_param, feature_param, curr_dir);
                    
                    % Compute Features:
                    currFeatures_dir = dir([curr_dir filesep 'EEG_Features' filesep 'Rev_' curr_dataset_name '_Epoch*.mat']);
                    currFeatures_finished = cellfun(@(x) strsplit(x,{'Epoch','.mat'}),{currFeatures_dir.name},'un',0); currFeatures_finished = cellfun(@(x) str2num(x{2}),currFeatures_finished);
                    epochs_to_process = setdiff(1:size(EEG.data,3),currFeatures_finished);
                    if ~isempty(epochs_to_process)
                        %if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
                        fprintf(['\n ***************************** Starting Feature Computation ***************************** \n']);
                        tic; compute_features_compiled(EEG,curr_dir,curr_dataset_name,feature_param.feature_names,general_param.base_path); toc
                    else
                        fprintf(['\n ***************************** Features Computed for All Epochs ***************************** \n']);
                    end

                    % Curate features:
                    fprintf(['\n ***************************** Curating Computed Features ***************************** \n']);
                    Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
                    Featurefiles_basename = ['Rev_' curr_dataset_name];
                    % [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);
                    [compute_feat] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 0);

                else
                    fprintf(['\n ********** CDT FILE MISSING :: Processing Subject: ' general_param.sub_dir_mod(kk).PID ', Run: ' curr_run ', Condition: ' curr_condition ' ********** \n']);
                end


            % catch e %dmer1
            % end %dmer1
        end
    end
end