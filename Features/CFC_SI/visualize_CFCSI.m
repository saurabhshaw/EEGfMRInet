
% Control Parameters:
ii = 1; % Subject Number
jj = 1; % Session Number
dataset_name = 'EEGfMRI_data';
feature_name = {'CFC_SI_mag'}; % Could be 'CFC_SI_mag', 'CFC_SI_theta', etc.

% Set Paths:
[base_path_rc, base_path_rd] = setPaths();
base_path_data = [base_path_rd filesep 'Analyzed_data' filesep 'EEG_fMRI_Combined_Dataset'];

curr_dir = base_path_data;
curr_file = [curr_dir filesep dataset_name '_Subject' num2str(ii) '_Session' num2str(jj) '.mat'];

loaded_data = load(curr_file, 'fMRI_labels_selected_window_thresh');
load([curr_dir filesep 'EEG_Features' filesep 'Rev_Sub' num2str(ii) '_Ses' num2str(jj) '_AllEpochs_' feature_name{1}]) % Can iterate through other features if stated in feature_name

% Create Label Vector:
mat_labels = loaded_data.fMRI_labels_selected_window_thresh;
% Modify output labels from multilabel to singlelabel (Find better way than this):
curr_labels_mRMR = zeros(size(mat_labels,1),1);
for i = 1:size(mat_labels,1)
    curr_values = find(mat_labels(i,:));
    if isempty(curr_values) curr_values = 0; end
    curr_labels_mRMR(i) = curr_values(1);    % Find a better way to do this for multi-label classification
end

% Read in the individual epochs based on the class labels:
