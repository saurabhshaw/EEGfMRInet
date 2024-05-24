%% generic settings
% Number of folds for cross-validation
numFolds = 10;
% Initialize variables to store accuracy for each fold
accuracy = zeros(numFolds, 1);
% load labels & aggregate
labels_rsEEG_Post_pDMN = (load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_rsEEG_Post_pDMN.mat').curr_labels_mRMR)';
labels_MInjury_pDMN = (load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_MInjury_pDMN.mat').curr_labels_mRMR)';
labels_Neutral_pDMN = (load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_Neutral_pDMN.mat').curr_labels_mRMR)';

save("Y_final_rsEEG_Post_pDMN.mat","labels_rsEEG_Post_pDMN");
save("Y_final_MInjury_pDMN.mat","labels_MInjury_pDMN");
save("Y_final_Neutral_pDMN.mat","labels_Neutral_pDMN");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cfc + coh + dpli + avgamp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get data
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1:3000);
final_features_rsEEG_Post_pDMN = final_feature_labels_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = [final_dataset_rsEEG_Post_pDMN final_dataset_mRMR];
final_features_rsEEG_Post_pDMN = [final_features_rsEEG_Post_pDMN final_feature_labels_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1:3000);
final_features_MInjury_pDMN = final_feature_labels_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = [final_dataset_MInjury_pDMN final_dataset_mRMR];
final_features_MInjury_pDMN = [final_features_MInjury_pDMN final_feature_labels_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1:3000);
final_features_Neutral_pDMN = final_feature_labels_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = [final_dataset_Neutral_pDMN final_dataset_mRMR];
final_features_Neutral_pDMN = [final_features_Neutral_pDMN final_feature_labels_mRMR];

save("X_final_rsEEG_Post_pDMN.mat","final_dataset_rsEEG_Post_pDMN");
save("X_final_MInjury_pDMN.mat","final_dataset_MInjury_pDMN");
save("X_final_Neutral_pDMN.mat","final_dataset_Neutral_pDMN");

save("X_final_feature_labels_rsEEG_Post_pDMN.mat","final_features_rsEEG_Post_pDMN");
save("X_final_feature_labels_MInjury_pDMN.mat","final_features_MInjury_pDMN");
save("X_final_feature_labels_Neutral_pDMN.mat","final_features_Neutral_pDMN");

%% pDMN: MINJURY
final_dataset_pDMN = final_dataset_MInjury_pDMN;
final_labels_pDMN = labels_MInjury_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN MInjury full model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN MInjury full (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN MInjury NULL full model: ' num2str(mean(accuracy))]);

%% pDMN: NEUTRAL - full model
final_dataset_pDMN = final_dataset_Neutral_pDMN;
final_labels_pDMN = labels_Neutral_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN Neutral full model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN Neutral full (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN Neutral NULL full model: ' num2str(mean(accuracy))]);

%% pDMN: RESTINGSTATEPOST - full model
final_dataset_pDMN = final_dataset_rsEEG_Post_pDMN;
final_labels_pDMN = labels_rsEEG_Post_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN rsEEG_Post full model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN rsEEG_Post full (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN rsEEG_Post NULL full model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cfc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1:1000);

%% pDMN: MINJURY
final_dataset_pDMN = final_dataset_MInjury_pDMN;
final_labels_pDMN = labels_MInjury_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN MInjury cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN MInjury cfc (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN MInjury NULL cfc model: ' num2str(mean(accuracy))]);

%% pDMN: NEUTRAL - cfc model
final_dataset_pDMN = final_dataset_Neutral_pDMN;
final_labels_pDMN = labels_Neutral_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN Neutral cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN Neutral cfc (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN Neutral NULL cfc model: ' num2str(mean(accuracy))]);

%% pDMN: RESTINGSTATEPOST - cfc model
final_dataset_pDMN = final_dataset_rsEEG_Post_pDMN;
final_labels_pDMN = labels_rsEEG_Post_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN rsEEG_Post cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN rsEEG_Post cfc (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN rsEEG_Post NULL cfc model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dpli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

final_dataset_rsEEG_Post_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat').final_dataset_mRMR;
final_dataset_MInjury_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat').final_dataset_mRMR;
final_dataset_Neutral_pDMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat').final_dataset_mRMR;

%% pDMN: MINJURY
final_dataset_pDMN = final_dataset_MInjury_pDMN;
final_labels_pDMN = labels_MInjury_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN MInjury dpli model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN MInjury dpli (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN MInjury NULL dpli model: ' num2str(mean(accuracy))]);

%% pDMN: NEUTRAL - dpli model
final_dataset_pDMN = final_dataset_Neutral_pDMN;
final_labels_pDMN = labels_Neutral_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN Neutral dpli model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN Neutral dpli (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN Neutral NULL dpli model: ' num2str(mean(accuracy))]);

%% pDMN: RESTINGSTATEPOST - dpli model
final_dataset_pDMN = final_dataset_rsEEG_Post_pDMN;
final_labels_pDMN = labels_rsEEG_Post_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN rsEEG_Post dpli model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN rsEEG_Post dpli (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN rsEEG_Post NULL dpli model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cfc+dpli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% pDMN: MINJURY
final_dataset_pDMN = final_dataset_MInjury_pDMN;
final_labels_pDMN = labels_MInjury_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN MInjury dpli+cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN MInjury dpli+cfc (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN MInjury NULL dpli+cfc model: ' num2str(mean(accuracy))]);

%% pDMN: NEUTRAL - dpli+cfc model
final_dataset_pDMN = final_dataset_Neutral_pDMN;
final_labels_pDMN = labels_Neutral_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN Neutral dpli+cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN Neutral dpli+cfc (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN Neutral NULL dpli+cfc model: ' num2str(mean(accuracy))]);

%% pDMN: RESTINGSTATEPOST - dpli+cfc model
final_dataset_pDMN = final_dataset_rsEEG_Post_pDMN;
final_labels_pDMN = labels_rsEEG_Post_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN rsEEG_Post dpli+cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN rsEEG_Post dpli+cfc (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN rsEEG_Post NULL dpli+cfc model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% arr_avgamp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,1001:2000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,1001:2000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,1001:2000);

%% pDMN: MINJURY
final_dataset_pDMN = final_dataset_MInjury_pDMN;
final_labels_pDMN = labels_MInjury_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN MInjury arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN MInjury arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN MInjury NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% pDMN: NEUTRAL - arr_avgamp model
final_dataset_pDMN = final_dataset_Neutral_pDMN;
final_labels_pDMN = labels_Neutral_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN Neutral arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN Neutral arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN Neutral NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% pDMN: RESTINGSTATEPOST - arr_avgamp model
final_dataset_pDMN = final_dataset_rsEEG_Post_pDMN;
final_labels_pDMN = labels_rsEEG_Post_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN rsEEG_Post arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN rsEEG_Post arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN rsEEG_Post NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% coh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_rsEEG_Post_pDMN = final_dataset_mRMR(:,2001:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_MInjury_pDMN = final_dataset_mRMR(:,2001:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_postDMN_mivo.mat')
final_dataset_Neutral_pDMN = final_dataset_mRMR(:,2001:3000);

%% pDMN: MINJURY
final_dataset_pDMN = final_dataset_MInjury_pDMN;
final_labels_pDMN = labels_MInjury_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN MInjury arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN MInjury arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN MInjury NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% pDMN: NEUTRAL - arr_avgamp model
final_dataset_pDMN = final_dataset_Neutral_pDMN;
final_labels_pDMN = labels_Neutral_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN Neutral arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN Neutral arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN Neutral NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% pDMN: RESTINGSTATEPOST - arr_avgamp model
final_dataset_pDMN = final_dataset_rsEEG_Post_pDMN;
final_labels_pDMN = labels_rsEEG_Post_pDMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_pDMN, 1), numFolds);
% training/validation, FULL
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
disp(['Average Accuracy, pDMN rsEEG_Post arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, pDMN rsEEG_Post arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

% training/validation, NULL
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
disp(['Average Accuracy, pDMN rsEEG_Post NULL arr_avgamp model: ' num2str(mean(accuracy))]);

