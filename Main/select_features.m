function select_features(workspace_filepath,roi_idx)
% get curated features
%% load participant-run-condition workspace (using 1083 as testcase)
load(workspace_filepath);
% load('mrmr_ready_1083MInjury.mat');
max_features = feature_param.max_features;
Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

%% load fmri conn data and ppt-mapping (add to setup params if possible)
% load('D:\research_data\HCWMIVO\all_data\ML_y\fMRI_ICNActivity_All.mat');
% find ppt index for data sync
ppt_idx = find(cellfun(@(x)isequal(x,participant_id),P_ID));
% get data subset as per idx (*iterate through rois)
roi_name = ROI_ID(roi_idx);
curr_fmri_data = data_curated(roi_idx,:,ppt_idx);

%% get feature labels for curr condition
curr_fmri_data_norm = normalize(curr_fmri_data,"norm"); %L2 p-norm. what's the best way to normalize?
activation_threshold = prctile(curr_fmri_data_norm,70); %threshold of 70th percentile -- how to be optimally conservative?
% get all labels for all conditions
curr_labels_mRMR_all = (curr_fmri_data_norm>activation_threshold)';
% reduce labels focusing only on current condition
curr_condition_timepoints = CONN_param.condition_sequence_w_timepoints(curr_condition);
all_conditions = keys(CONN_param.condition_sequence_w_timepoints);
curr_condition_start = 0;
for i = 1:size(all_conditions)
    condition = all_conditions(i);
    if condition == curr_condition
        break
    else
        curr_condition_start = curr_condition_start + CONN_param.condition_sequence_w_timepoints(condition);
    end
end
curr_condition_end = curr_condition_start+curr_condition_timepoints;
% final labels
curr_labels_mRMR = double(curr_labels_mRMR_all(curr_condition_start+1:curr_condition_end));

%% iterative mRMR
%initializations****
wind_dataset_mRMR = []; wind_feature_labels_mRMR = [];
wind_output_features = cell(1,length(currFeatures_curated)); wind_output_scores = cell(1,length(currFeatures_curated));

for i = 1:length(currFeatures_curated)
    % load feature set
    feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
    % convert feature set from linear to n-dimensional
    curr_Feature_curated = cellfun(@(x)reshape(x,[feat_file.Feature_size]),feat_file.Feature,'un',0);
    % get feature sizes
    innermost_feature_size = feat_file.Feature_size(end-1:end);
    feat_file_Feature_size = feat_file.Feature_size;
    if length(feat_file_Feature_size)==4 % Single Frequency Computation
        % initializations per window****
        freq_dataset_mRMR = cell(1,feat_file_Feature_size(1)); freq_feature_labels_mRMR = cell(1,feat_file_Feature_size(1));
        freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
        % parfor m = 1:feat_file_Feature_size(1) % Number of Windows
        for m = 1:2 % replace me with line above -- TESTING
            % initializations per frequency****
            elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
            elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
            for n = 1:feat_file_Feature_size(2) % Number of Frequency Windows
                % get current x-dataset: all timepoints @ timepoint-window m, frequency n
                curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,n,:,:)),curr_Feature_curated,'un',0);% squeeze -> from dim 1x1x68x68 to dim 68x68 cell, for all timepoints
                curr_dataset_mRMR = cellfun(@(x)x(:),curr_dataset_mRMR,'un',0);% converts dim 68x68 cell into dim 4624x1 cell column vector, for all timepoints
                curr_dataset_mRMR = cell2mat(curr_dataset_mRMR);% converts (merges) dim 4624x1 cell column vector of each timepoint together into single 4624x118 matrix  
                curr_dataset_mRMR = curr_dataset_mRMR';%transposes said matrix for correct dimensionality for mRMR x input -- ready for mrmr x-input
                
                % run mRMR: all timepoints of curr condition @ timepoint-window m, frequency n
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,innermost_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR);

                disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBand ' num2str(n)]);
            end

            curr_feature_size = [feat_file_Feature_size(2) size(elec_output_features{1},2)];
            [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR{m},freq_feature_labels_mRMR{m}] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR{m},freq_feature_labels_mRMR{m},elec_feature_labels_mRMR);

        end

        freq_dataset_mRMR = cat(2,freq_dataset_mRMR{:});
        freq_feature_labels_mRMR = cat(2,freq_feature_labels_mRMR{:});

    else % Between frequency computation
        freq_dataset_mRMR = []; freq_feature_labels_mRMR = [];
        freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
        % for m = 1:feat_file_Feature_size(1) % Number of Windows
        for m = 1:2 % replace me with line above -- TESTING
            
            % elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
            elec_dataset_mRMR = cell(1,feat_file_Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file_Feature_size(2));
            elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
            
            curr_dataset_mRMR_parfor = cellfun(@(x)squeeze(x(m,:,:,:,:)),curr_Feature_curated,'un',0);

            tic
            % Run one less outer loop for frequency due to the n < p
            % computation used for cross-frequency features - this causes
            % all n-p combinations of the last n to be empty
            parfor n = 1:(feat_file_Feature_size(2)-1) % Number of Frequency Windows
                
                innerFreq_dataset_mRMR = []; innerFreq_feature_labels_mRMR = []; innerFreq_loopIDX_mRMR = [];
                innerFreq_output_features = cell(1,feat_file_Feature_size(3)); innerFreq_output_scores = cell(1,feat_file_Feature_size(3));

                for p = 1:feat_file_Feature_size(3)
                    % flip p and n index positions because this was not
                    % kept consistent from the single frequency computation
                    % where n is the second index - fix this by indexing
                    % curr_Feature_curated(m,n,p,:,:) in
                    % curate_features_deploy in future
                    % But since the number of features on both levels are
                    % the same - n,p works
                    
                    if n < p % This is how features were computed for cross-frequency features
                        % Get current dataset for mRMR:
                        curr_dataset_mRMR = cellfun(@(x)squeeze(x(n,p,:,:)),curr_dataset_mRMR_parfor,'un',0);
                        curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
                        
                        % Run mRMR at this level:
                        [innerFreq_output_features{p}, innerFreq_output_scores{p},innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,innermost_feature_size,max_features,innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR);                        
                        innerFreq_loopIDX_mRMR = [innerFreq_loopIDX_mRMR repmat([p],1,size(innerFreq_output_features{p},2))];
                        
                    end
                    disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBands ' num2str(n) '-' num2str(p)]);
                    
                end
                
                % Run mRMR at this level:

                curr_feature_size = [feat_file_Feature_size(3) size(innerFreq_output_features{end},2)];
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_iterate_loop(innerFreq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n},innerFreq_feature_labels_mRMR,innerFreq_loopIDX_mRMR);
                
            end
            toc
            
            % Run mRMR at this level:
            elec_loopIDX_mRMR = [];
            for temp_idx = 1:length(elec_output_features) elec_loopIDX_mRMR = [elec_loopIDX_mRMR repmat([temp_idx],[1 size(elec_output_features{temp_idx},2)])]; end
            
            elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
            elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
            curr_feature_size = [feat_file_Feature_size(2) size(elec_output_features{1},2)];
            [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR,freq_feature_labels_mRMR] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,freq_feature_labels_mRMR,elec_feature_labels_mRMR,elec_loopIDX_mRMR);
            
        end
    end
    % Run mRMR at this level:
    curr_feature_size = [feat_file_Feature_size(1) size(freq_output_features{1},2)];
    [wind_output_features{i}, wind_output_scores{i},wind_dataset_mRMR,wind_feature_labels_mRMR] = mRMR_iterate_loop(freq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,wind_dataset_mRMR,wind_feature_labels_mRMR,freq_feature_labels_mRMR);

end

final_dataset_mRMR = wind_dataset_mRMR;
final_feature_labels_mRMR = wind_feature_labels_mRMR;

nestedFeature_struct = [];
nestedFeature_struct.wind_output_features = wind_output_features;
nestedFeature_struct.wind_output_scores = wind_output_scores;
nestedFeature_struct.freq_output_features = freq_output_features;
nestedFeature_struct.freq_output_scores = freq_output_scores;
nestedFeature_struct.freq_dataset_mRMR = freq_dataset_mRMR;
nestedFeature_struct.freq_feature_labels_mRMR = freq_feature_labels_mRMR;

save_path = [Featurefiles_directory filesep 'mrmr' filesep Featurefiles_basename '_mRMRiterateResults_' roi_name];
save(save_path,'final_dataset_mRMR','final_feature_labels_mRMR','currFeatures_curated','nestedFeature_struct');
