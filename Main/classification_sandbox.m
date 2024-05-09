%% generic settings
% Number of folds for cross-validation
numFolds = 10;
% Initialize variables to store accuracy for each fold
accuracy = zeros(numFolds, 1);

%% SMN - full model training with linear kernel SVM
% load training data & aggregate
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = [final_dataset_rsEEG_Post_SMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = [final_dataset_MInjury_SMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = [final_dataset_Neutral_SMN final_dataset_mRMR];
final_dataset_SMN = [final_dataset_Neutral_SMN;final_dataset_MInjury_SMN;final_dataset_rsEEG_Post_SMN];% keep note order for label vector
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);

% load labels & aggregate
labels_rsEEG_Post_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_rsEEG_Post_SMN.mat').curr_labels_mRMR;
labels_MInjury_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_MInjury_SMN.mat').curr_labels_mRMR;
labels_Neutral_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_Neutral_SMN.mat').curr_labels_mRMR;
final_labels_SMN = [labels_Neutral_SMN labels_MInjury_SMN labels_rsEEG_Post_SMN]';% keep note order

% training/validation, FULL
for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_full = accuracy;
svmModel_SMN_full = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN full model: ' num2str(mean(accuracy))]);

% training/validation, FULL, tested on training data
for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, trainData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == trainLabels) / numel(trainLabels);
end

accuracy_SMN_full_testedOnTrained = accuracy;
svmModel_SMN_full_testedOnTrained = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN full (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);trainLabels = trainLabels(randperm(length(trainLabels)));%randomizing
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_null = accuracy;
svmModel_SMN_null = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN NULL model: ' num2str(mean(accuracy))]);

% training/validation, cfc
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1:1000);
final_dataset_SMN = [final_dataset_Neutral_SMN;final_dataset_MInjury_SMN;final_dataset_rsEEG_Post_SMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_cfc = accuracy;
svmModel_SMN_cfc = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN cfc model: ' num2str(mean(accuracy))]);

% training/validation, avgamp
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1001:2000);
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1001:2000);
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1001:2000);
final_dataset_SMN = [final_dataset_Neutral_SMN;final_dataset_MInjury_SMN;final_dataset_rsEEG_Post_SMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_avgamp = accuracy;
svmModel_SMN_avgamp = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN avgamp model: ' num2str(mean(accuracy))]);

% training/validation, coh
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,2001:3000);
final_dataset_MInjury_SMN = final_dataset_mRMR(:,2001:3000);
final_dataset_Neutral_SMN = final_dataset_mRMR(:,2001:3000);
final_dataset_SMN = [final_dataset_Neutral_SMN;final_dataset_MInjury_SMN;final_dataset_rsEEG_Post_SMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_coh = accuracy;
svmModel_SMN_coh = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN coh model: ' num2str(mean(accuracy))]);

% training/validation, dpli
final_dataset_rsEEG_Post_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat').final_dataset_mRMR;
final_dataset_MInjury_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat').final_dataset_mRMR;
final_dataset_Neutral_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat').final_dataset_mRMR;
final_dataset_SMN = [final_dataset_Neutral_SMN;final_dataset_MInjury_SMN;final_dataset_rsEEG_Post_SMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_dpli = accuracy;
svmModel_SMN_dpli = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN dpli model: ' num2str(mean(accuracy))]);

% training/validation, cfc + dpli
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = [final_dataset_rsEEG_Post_SMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = [final_dataset_MInjury_SMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = [final_dataset_Neutral_SMN final_dataset_mRMR];
final_dataset_SMN = [final_dataset_Neutral_SMN;final_dataset_MInjury_SMN;final_dataset_rsEEG_Post_SMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_SMN(trainIdx, :);
    testData = final_dataset_SMN(testIdx, :);
    trainLabels = final_labels_SMN(trainIdx);
    testLabels = final_labels_SMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_SMN_dpli_cfc = accuracy;
svmModel_SMN_dpli_cfc = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, SMN dpli+cfc model: ' num2str(mean(accuracy))]);

%% pDMN
% load training data & aggregate
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = [final_dataset_rsEEG_Post_pDMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = [final_dataset_MInjury_pDMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = [final_dataset_Neutral_pDMN final_dataset_mRMR];
final_dataset_pDMN = [final_dataset_Neutral_pDMN;final_dataset_MInjury_pDMN;final_dataset_rsEEG_Post_pDMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);

% load labels & aggregate
labels_rsEEG_Post_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_rsEEG_Post_pDMN.mat').curr_labels_mRMR;
labels_MInjury_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_MInjury_pDMN.mat').curr_labels_mRMR;
labels_Neutral_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_Neutral_pDMN.mat').curr_labels_mRMR;
final_labels_pDMN = [labels_Neutral_pDMN labels_MInjury_pDMN labels_rsEEG_Post_pDMN]';% keep note order to align to training data

% training/validation, full
for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_full = accuracy;
svmModel_pDMN_full = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN full model: ' num2str(mean(accuracy))]);

% training/validation, FULL, tested on training data
for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, trainData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == trainLabels) / numel(trainLabels);
end

accuracy_pDMN_full_testedOnTrained = accuracy;
svmModel_pDMN_full_testedOnTrained = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN full (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, null
for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);trainLabels = trainLabels(randperm(length(trainLabels)));%randomizing
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_null = accuracy;
svmModel_pDMN_null = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN null model: ' num2str(mean(accuracy))]);

% training/validation, cfc
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1:1000);
final_dataset_pDMN = [final_dataset_Neutral_pDMN;final_dataset_MInjury_pDMN;final_dataset_rsEEG_Post_pDMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_cfc = accuracy;
svmModel_pDMN_cfc = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN cfc model: ' num2str(mean(accuracy))]);

% training/validation, avgamp
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1001:2000);
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1001:2000);
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1001:2000);
final_dataset_pDMN = [final_dataset_Neutral_pDMN;final_dataset_MInjury_pDMN;final_dataset_rsEEG_Post_pDMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_avgamp = accuracy;
svmModel_pDMN_avgamp = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN avgamp model: ' num2str(mean(accuracy))]);

% training/validation, coh
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,2001:3000);
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,2001:3000);
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,2001:3000);
final_dataset_pDMN = [final_dataset_Neutral_pDMN;final_dataset_MInjury_pDMN;final_dataset_rsEEG_Post_pDMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_coh = accuracy;
svmModel_pDMN_coh = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN coh model: ' num2str(mean(accuracy))]);

% training/validation, dpli
final_dataset_rsEEG_Post_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat').final_dataset_mRMR;
final_dataset_MInjury_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat').final_dataset_mRMR;
final_dataset_Neutral_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat').final_dataset_mRMR;
final_dataset_pDMN = [final_dataset_Neutral_pDMN;final_dataset_MInjury_pDMN;final_dataset_rsEEG_Post_pDMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_dpli = accuracy;
svmModel_pDMN_dpli = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN dpli model: ' num2str(mean(accuracy))]);

% training/validation, dpli+cfc
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = [final_dataset_rsEEG_Post_pDMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = [final_dataset_MInjury_pDMN final_dataset_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = [final_dataset_Neutral_pDMN final_dataset_mRMR];
final_dataset_pDMN = [final_dataset_Neutral_pDMN;final_dataset_MInjury_pDMN;final_dataset_rsEEG_Post_pDMN];% keep note order for label vector
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);

for i = 1:numFolds
    % Split data into training and testing sets
    testIdx = (indices == i);
    trainIdx = ~testIdx;
    trainData = final_dataset_pDMN(trainIdx, :);
    testData = final_dataset_pDMN(testIdx, :);
    trainLabels = final_labels_pDMN(trainIdx);
    testLabels = final_labels_pDMN(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'linear');
    
    % Predict labels for test data
    predictedLabels = predict(svmModel, testData);
    
    % Calculate accuracy
    accuracy(i) = sum(predictedLabels == testLabels) / numel(testLabels);
end

accuracy_pDMN_dpli_cfc = accuracy;
svmModel_pDMN_dpli_cfc = svmModel;
% Display average accuracy across all folds
disp(['Average Accuracy, pDMN dpli+cfc model: ' num2str(mean(accuracy))]);
