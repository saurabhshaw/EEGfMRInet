% function [curr_Features, curr_YY_final] = eegnet_getFeatures(CONN_data,CONN_cfg,runs_to_include,output_base_path_data,sub_dir,sub_dir_mod,EEGfMRI_corrIDX,classifierType,varargin)


%% Create Group Matfile Array:
% sub_dir; sub_dir_mod; output_base_path_data; EEGfMRI_corrIDX; CONN_data
jj = 1;

Featurefiles_directory_ALL = []; Featurefiles_basename_ALL = []; curr_Featurefiles_basename_ALL = [];
YY_final_ALL = []; YY_final_continuous_ALL = []; subIDX_ALL = []; sessIDX_ALL = [];
for ii = 1:length(sub_dir)
    skip_analysis = 0;
    dataset_to_use = [sub_dir(ii).name];
    dataset_name = [sub_dir_mod(ii).PID];
    curr_dir = [output_base_path_data filesep dataset_to_use];
    
    temp_idx = ~isnan(EEGfMRI_corrIDX(ii,:)); % Check if this task block has been processed by CONN:

    for m = find(temp_idx)
        
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
YY_final_All_sessIDX = cellfun(@(x,y) repmat(y,[1 length(x)]),YY_final_ALL,mat2cell(sessIDX_ALL,ones(size(sessIDX_ALL))),'un',0);

Featurefiles_curated_dir = dir([Featurefiles_directory_ALL{1} filesep curr_Featurefiles_basename_ALL{1} '_AllEpochs_*.mat']);
% Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

%% Get the Features:
if isempty(varargin) % Load features from pre-existing files:
    
    % Create output directory if not already made:
    % features_to_include = [1 2 3 4 5];
    features_to_include = 1:length(currFeatures_curated);
    
    curr_dir = [output_base_path_data filesep [sub_dir(end).name]];
    if isempty(dir(curr_dir))
        mkdir(curr_dir)
    end
    
    curr_labels_mRMR = cat(2,YY_final_ALL{:}); % Get current labels for mRMR
    curr_labels_mRMR_subIDX = cat(2,YY_final_All_subIDX{:}); % Get current labels for mRMR
    curr_labels_mRMR_sessIDX = cat(2,YY_final_All_sessIDX{:}); % Get current labels for mRMR
    curr_Features = [];
    for i = sort(features_to_include)
        curr_Features_struct = load([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep study_name '_' currFeatures_curated{i} '_mRMRiterateGroupResults_' CONN_cfg.class_types],'final_dataset_mRMR','curr_dataset_mRMR_IDX');
        curr_Feature_vect = nan(length(curr_labels_mRMR),size(curr_Features_struct.final_dataset_mRMR,2));
        curr_IDX = squeeze(curr_Features_struct.curr_dataset_mRMR_IDX);
        
        curr_Feature_vect(curr_IDX,:) = curr_Features_struct.final_dataset_mRMR;
        
        curr_Features = cat(2,curr_Features,curr_Feature_vect);
        
    end
    curr_YY_final = curr_labels_mRMR;
    curr_YY_final_continuous = cat(1,YY_final_continuous_ALL{:});
    
else % Load the specified features:
    
    % TO DO
    
end