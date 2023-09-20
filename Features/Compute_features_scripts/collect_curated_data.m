
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

%% Find curated features:
% curr_Featurefiles_basename = strsplit(Featurefiles_basename,'_CLASS');
% curr_Featurefiles_basename = curr_Featurefiles_basename{1};
Featurefiles_curated_dir = dir([Featurefiles_directory_ALL{1} filesep curr_Featurefiles_basename_ALL{1} '_AllEpochs_*.mat']);
% Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);


%% Collect the data:                
% curr_labels_mRMR = YY_final; % Get current labels for mRMR
curr_labels_mRMR = cat(2,YY_final_ALL{:}); % Get current labels for mRMR
IDX_mat = cumsum(cellfun(@(x) length(x),YY_final_ALL)); IDX_mat = [0; IDX_mat];

wind_dataset_mRMR = []; wind_feature_labels_mRMR = [];
wind_output_features = cell(1,length(currFeatures_curated)); wind_output_scores = cell(1,length(currFeatures_curated));

if isempty(dir([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name]))
    mkdir([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name]);
end

for i = 1:length(currFeatures_curated) % Number of features
    
    % Create matfile File:
    load([Featurefiles_directory_ALL{1} filesep curr_Featurefiles_basename_ALL{1} '_AllEpochs_' currFeatures_curated{i} '.mat'],'Feature_size');
    save([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'AllData_' currFeatures_curated{i}],'Feature_size','-v7.3');
    
    groupFeature_matfile = matfile([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'AllData_' currFeatures_curated{i}],'Writable',true);
    % groupFeature_matfile.Feature = cell(1,length(curr_Featurefiles_basename_ALL));
    groupFeature_matfile.Feature = cell(Feature_size(1),Feature_size(2),length(curr_labels_mRMR));
    
    % Transfer the data:
    %feat_file_matfiles = cellfun(@(x,y)matfile([x filesep y '_AllEpochs_' currFeatures_curated{i} '.mat']),Featurefiles_directory_ALL,curr_Featurefiles_basename_ALL,'un',0);
    for j = 1:length(curr_Featurefiles_basename_ALL)
        
        if ~isempty(dir([Featurefiles_directory_ALL{j} filesep curr_Featurefiles_basename_ALL{j} '_AllEpochs_' currFeatures_curated{i} '.mat']))
            
            tic;
            curr_files = load([Featurefiles_directory_ALL{j} filesep curr_Featurefiles_basename_ALL{j} '_AllEpochs_' currFeatures_curated{i} '.mat']);
            curr_Feature_curated = cellfun(@(x)reshape(x,[curr_files.Feature_size]),curr_files.Feature,'un',0);
            % groupFeature_matfile.Feature(j) = {curr_files.Feature};
            num_timepnts = length(curr_Feature_curated);
            temp_feature = cat(length(curr_files.Feature_size) + 1,curr_Feature_curated{:});
            
            if length(Feature_size) == 4
                temp_feature = permute(temp_feature,[1 2 length(Feature_size)+1 3 4]);
                temp_feature = mat2cell(temp_feature,ones(1,Feature_size(1)),ones(1,Feature_size(2)),ones(1,num_timepnts),Feature_size(end-1),Feature_size(end));
            else
                temp_feature = permute(temp_feature,[1 2 length(Feature_size)+1 3 4 5]);
                temp_feature = mat2cell(temp_feature,ones(1,Feature_size(1)),ones(1,Feature_size(2)),ones(1,num_timepnts),Feature_size(end-2),Feature_size(end-1),Feature_size(end));
            end
            
            temp_feature = cellfun(@(x)squeeze(x),temp_feature,'un',0);
            startIDX = IDX_mat(j) + 1; endIDX = IDX_mat(j+1);
            if (endIDX-startIDX+1) > num_timepnts
                endIDX = endIDX - 1;
            end
            % groupFeature_matfile.Feature(:,:,startIDX:endIDX) = cat(length(curr_files.Feature_size) + 1,curr_Feature_curated{:});
            groupFeature_matfile.Feature(1:Feature_size(1),1:Feature_size(2),startIDX:endIDX) = temp_feature;
            toc
            
            disp(['Finished ' num2str(j) '/' num2str(length(curr_Featurefiles_basename_ALL))]);
            
        end
    end
    
end