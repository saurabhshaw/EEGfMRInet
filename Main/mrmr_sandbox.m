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

%% obtain label vectors (activation)

activation_threshold = 0.3; %how to properly quantify binary activation/non-activation of fmri time series?

%smn signal intensity labels
smn_sig_intensity = data_curated(1,:,ppt_no);%for first participant
%normalize signal intensities
smn_sig_intensity_norm = normalize(smn_sig_intensity);%normalization method: zcore (default) 
%get activation labels
smn_y = (smn_sig_intensity_norm>activation_threshold)';

%dmn signal intensity labels
dmn_sig_intensity = data_curated(2,:,ppt_no);%for first participant
%normalize signal intensities
dmn_sig_intensity_norm = normalize(dmn_sig_intensity);%normalization method: zcore (default)
%get activation labels
dmn_y = (dmn_sig_intensity_norm>activation_threshold)';

%% Remove redundant features using upper triang for each timepoint and merge into full feature set
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

%% TESTING %%
% [TEST_1_smn_feature_rank, TEST_1_smn_rank_scores] = fscmrmr(final_features,smn_y(160)); 
% [TEST_1_dmn_feature_rank, TEST_1_dmn_rank_scores] = fscmrmr(final_features,dmn_y(160));
tic
% [TEST_2_smn_feature_rank, TEST_2_smn_rank_scores] = fscmrmr(final_features_all_timepoints(:,1:1000),smn_y(1:160)); 
[TEST_2_dmn_feature_rank, TEST_2_dmn_rank_scores] = fscmrmr(final_features_all_timepoints(:,1:3000),dmn_y(1:160));
toc

tic
% [TEST_2_smn_feature_rank, TEST_2_smn_rank_scores] = fscmrmr(final_features_all_timepoints(:,1:1000),smn_y(1:160)); 
[TEST_X_dmn_feature_rank, TEST_X_dmn_rank_scores] = mRMR(final_features_all_timepoints(:,1:3000),double(dmn_y(1:160)),3000);
toc
% display(isequal(TEST_1_dmn_feature_rank,TEST_2_dmn_feature_rank));
% display(isequal(TEST_1_smn_feature_rank,TEST_2_smn_feature_rank));