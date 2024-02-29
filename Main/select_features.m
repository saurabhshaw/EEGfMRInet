% get curated features
%% load participant-run-condition workspace (using 1083 as testcase)
load('mrmr_ready_1083MInjury.mat');
max_features = feature_param.max_features;
Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

%% load fmri conn data and ppt-mapping (added to setup params if possible)
load('D:\research_data\HCWMIVO\all_data\ML_y\fMRI_ICNActivity_All.mat');
% find ppt index for data sync
ppt_idx = find(cellfun(@(x)isequal(x,participant_id),P_ID));
% get data subset as per idx (*iterate through rois)
curr_fmri_data = data_curated(1,:,ppt_idx);

%% get feature labels for curr condition
activation_threshold = CONN_param.threshold; %define a good threshold
curr_fmri_data_norm = normalize(curr_fmri_data); %what's the best way to normalize? defaulting zscore, pnorm doesnt seem to work - TS.
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
        display(curr_condition_start);
    end
end
curr_condition_end = curr_condition_start+curr_condition_timepoints;
% final labels
curr_labels_mRMR = curr_labels_mRMR_all(curr_condition_start+1:curr_condition_end);

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
    if length(feat_file_Feature_size)==4
        % initializations per window****
        freq_dataset_mRMR = cell(1,feat_file_Feature_size(1)); freq_feature_labels_mRMR = cell(1,feat_file_Feature_size(1));
        freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
        for m = 1:feat_file_Feature_size(1)
            % initializations per frequency****
            elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
            elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
            for n = 1:feat_file_Feature_size(2)
                % get current x-dataset: all timepoints @ timepoint-window m, frequency n
                curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,n,:,:)),curr_Feature_curated,'un',0);% squeeze -> from dim 1x1x68x68 to dim 68x68 cell, for all timepoints
                curr_dataset_mRMR = cellfun(@(x)x(:),curr_dataset_mRMR,'un',0);% converts dim 68x68 cell into dim 4624x1 cell column vector, for all timepoints
                curr_dataset_mRMR = cell2mat(curr_dataset_mRMR);% converts (merges) dim 4624x1 cell column vector of each timepoint together into single 4624x118 matrix  
                curr_dataset_mRMR = curr_dataset_mRMR';%transposes said matrix for correct dimensionality for mRMR x input -- ready for mrmr x-input
                % run mRMR: all timepoints of curr condition @ timepoint-window m, frequency n
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,innermost_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR);

                disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBand ' num2str(n)]);
            end

        end
    end

end


