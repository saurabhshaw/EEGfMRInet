%% read in raw data

% load feature data
folder_path_x = "D:\research_data\HWCMIVO\dataset_1\20210426_HCWMIVO_EEGfMRI-1005HCW\EEG_Features\";
filename_x = "Rev_rest_1005HCW_rsEEG_Post_AllEpochs_avg_coherence.mat";
fp_x = folder_path_x + filename_x;
load(fp_x)

% load fmri activation data
folder_path_y = "D:\research_data\HWCMIVO\ML_y\";
filename_y = "fMRI_ICNActivity_All.mat";
fp_y = folder_path_y + filename_y;
load(fp_y)

%% obtain label vector
% split into SMN & DMN activation (just for organization)
activation_threshold = 0.3; %how to properly quantify binary activation/non-activation of fmri time series?

%smn activity labels (for first participant)
smn_activity_cont = data_curated(1,:,1);%for first participant
smn_activity_bin = smn_activity_cont>activation_threshold;

%dmn activity labels
dmn_activity_cont = data_curated(2,:,1);%for first participant
dmn_activity_bin = dmn_activity_cont>activation_threshold;

%% sync X and Y: participants
%

%% prepare to remove redundant features
if ndims(curr_Feature_curated)==4
    step_size = Feature_size(1)*Feature_size(2);
elseif ndims(curr_Feature_curated)==5
    step_size = Feature_size(1)*Feature_size(2)*Feature_size(3);
end

new_col_vec = [];
for i = 1:step_size:length(Feature{1,1})-(step_size-1)
    new_block =  {Feature{1,1}(i:i+(step_size-1))};
    new_col_vec = [new_col_vec new_block];
end 
feature_cell = reshape(new_col_vec,68,68);
feature_mat = cell2mat(feature_cell);

%% get upper triag and reshape feature var
features = [];
iteration = 1;
for col = 2:size(feature_mat,2)
   display(col);
   display(iteration);
   new_block = feature_mat(1:step_size*iteration,col);
   features = [features new_block'];
   iteration=iteration+1;
end



