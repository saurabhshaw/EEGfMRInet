%% read in raw data
% select participant
ppt_no = 1; %to iterate over all participants
% feature_of_interest = 'avg_coherence';

% get filepaths
data_fp = "D:\research_data\HWCMIVO\all_data\";
data_fp_x = data_fp + "ML_x\";
data_fp_y = data_fp + "ML_y\";

% load fmri activation data
fmri_signal_intensities = "fMRI_ICNActivity_All.mat";
fp_y = data_fp_y + fmri_signal_intensities;
load(fp_y)

% load ppt-specific feature data
% ppt_id = P_ID(ppt_no);
folder_path_x = "D:\research_data\HWCMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\";
eeg_features = "Rev_rest_1001_rsEEG_Post_AllEpochs_avg_coherence.mat";%to iterate over all features
num_timepoints = 160;
fp_x = folder_path_x + eeg_features;
load(fp_x)


%% obtain label vector
% split into SMN & DMN activation (just for organization)
activation_threshold = 0.1; %how to properly quantify binary activation/non-activation of fmri time series?

%smn activity labels (for first participant)
smn_activity_cont = data_curated(1,:,ppt_no);%for first participant
smn_y = (smn_activity_cont>activation_threshold)';

%dmn activity labels
dmn_activity_cont = data_curated(2,:,ppt_no);%for first participant
dmn_y = (dmn_activity_cont>activation_threshold)';

%% Remove redundant features using upper/lower triang for each timepoint and merge into full feature set
% get dimension specifics depending on feature computed
if ndims(curr_Feature_curated)==4
    step_size = Feature_size(1)*Feature_size(2);
elseif ndims(curr_Feature_curated)==5
    step_size = Feature_size(1)*Feature_size(2)*Feature_size(3);
end

final_features_all_timepoints = [];
for feature_timepoint = 1:num_timepoints
    % shape data to allow for triangulation
    new_col_vec = [];
    for i = 1:step_size:length(Feature{1,feature_timepoint})-(step_size-1)
        new_block =  {Feature{1,feature_timepoint}(i:i+(step_size-1))};
        new_col_vec = [new_col_vec new_block];
    end 
    feature_cell = reshape(new_col_vec,68,68);
    feature_mat = cell2mat(feature_cell);
    
    % get upper triag and reshape for x (rows:observation@timepoint,col:feature) -> y (column vector:activation values)
    final_features = [];
    iteration = 1;
    for col = 2:size(feature_mat,2)
       new_block = feature_mat(1:step_size*iteration,col);
       final_features = [final_features new_block'];
       iteration=iteration+1;
    end
    final_features_all_timepoints = [final_features_all_timepoints;final_features];
end

%% get feature ranks and scores

%using fscmrmr for all timepoints
% [smn_feature_rank, smn_rank_scores] = fscmrmr(final_features_all_timepoints,smn_y(1:num_timepoints)); 
% [dmn_feature_rank, dmn_rank_scores] = fscmrmr(final_features_all_timepoints,dmn_y(1:num_timepoints));

%test - using fscmrmr for single timepoint
[TEST_smn_feature_rank, TEST_smn_rank_scores] = fscmrmr(final_features,smn_y(160)); 
[TEST_dmn_feature_rank, TEST_dmn_rank_scores] = fscmrmr(final_features,dmn_y(160));
% display(isequal(dmn_feature_rank,TEST_dmn_feature_rank));
