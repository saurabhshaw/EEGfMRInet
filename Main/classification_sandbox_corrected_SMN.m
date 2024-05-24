%% generic settings
% Number of folds for cross-validation
numFolds = 10;
% Initialize variables to store accuracy for each fold
accuracy = zeros(numFolds, 1);
% load labels & aggregate
labels_rsEEG_Post_SMN = (load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_rsEEG_Post_SMN.mat').curr_labels_mRMR)';
labels_MInjury_SMN = (load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_MInjury_SMN.mat').curr_labels_mRMR)';
labels_Neutral_SMN = (load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\labels\curr_labels_Neutral_SMN.mat').curr_labels_mRMR)';

save("Y_final_rsEEG_Post_SMN.mat","labels_rsEEG_Post_SMN");
save("Y_final_MInjury_SMN.mat","labels_MInjury_SMN");
save("Y_final_Neutral_SMN.mat","labels_Neutral_SMN");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cfc + coh + dpli + avgamp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get data
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1:3000);
final_features_rsEEG_Post_SMN = final_feature_labels_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = [final_dataset_rsEEG_Post_SMN final_dataset_mRMR];
final_features_rsEEG_Post_SMN = [final_features_rsEEG_Post_SMN final_feature_labels_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1:3000);
final_features_MInjury_SMN = final_feature_labels_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = [final_dataset_MInjury_SMN final_dataset_mRMR];
final_features_MInjury_SMN = [final_features_MInjury_SMN final_feature_labels_mRMR];
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1:3000);
final_features_Neutral_SMN = final_feature_labels_mRMR(:,1:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = [final_dataset_Neutral_SMN final_dataset_mRMR];
final_features_Neutral_SMN = [final_features_Neutral_SMN final_feature_labels_mRMR];

save("X_final_rsEEG_Post_SMN.mat","final_dataset_rsEEG_Post_SMN");
save("X_final_MInjury_SMN.mat","final_dataset_MInjury_SMN");
save("X_final_Neutral_SMN.mat","final_dataset_Neutral_SMN");

save("X_final_feature_labels_rsEEG_Post_SMN.mat","final_features_rsEEG_Post_SMN");
save("X_final_feature_labels_MInjury_SMN.mat","final_features_MInjury_SMN");
save("X_final_feature_labels_Neutral_SMN.mat","final_features_Neutral_SMN");

%% SMN: MINJURY
final_dataset_SMN = final_dataset_MInjury_SMN;
final_labels_SMN = labels_MInjury_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN MInjury full model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury full (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury NULL full model: ' num2str(mean(accuracy))]);

%% SMN: NEUTRAL - full model
final_dataset_SMN = final_dataset_Neutral_SMN;
final_labels_SMN = labels_Neutral_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN Neutral full model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral full (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral NULL full model: ' num2str(mean(accuracy))]);

%% SMN: RESTINGSTATEPOST - full model
final_dataset_SMN = final_dataset_rsEEG_Post_SMN;
final_labels_SMN = labels_rsEEG_Post_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN rsEEG_Post full model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post full (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post NULL full model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cfc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1:1000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1:1000);

%% SMN: MINJURY
final_dataset_SMN = final_dataset_MInjury_SMN;
final_labels_SMN = labels_MInjury_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN MInjury cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury cfc (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury NULL cfc model: ' num2str(mean(accuracy))]);

%% SMN: NEUTRAL - cfc model
final_dataset_SMN = final_dataset_Neutral_SMN;
final_labels_SMN = labels_Neutral_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN Neutral cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral cfc (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral NULL cfc model: ' num2str(mean(accuracy))]);

%% SMN: RESTINGSTATEPOST - cfc model
final_dataset_SMN = final_dataset_rsEEG_Post_SMN;
final_labels_SMN = labels_rsEEG_Post_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN rsEEG_Post cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post cfc (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post NULL cfc model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dpli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

final_dataset_rsEEG_Post_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat').final_dataset_mRMR;
final_dataset_MInjury_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat').final_dataset_mRMR;
final_dataset_Neutral_SMN = load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat').final_dataset_mRMR;

%% SMN: MINJURY
final_dataset_SMN = final_dataset_MInjury_SMN;
final_labels_SMN = labels_MInjury_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN MInjury dpli model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury dpli (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury NULL dpli model: ' num2str(mean(accuracy))]);

%% SMN: NEUTRAL - dpli model
final_dataset_SMN = final_dataset_Neutral_SMN;
final_labels_SMN = labels_Neutral_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN Neutral dpli model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral dpli (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral NULL dpli model: ' num2str(mean(accuracy))]);

%% SMN: RESTINGSTATEPOST - dpli model
final_dataset_SMN = final_dataset_rsEEG_Post_SMN;
final_labels_SMN = labels_rsEEG_Post_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN rsEEG_Post dpli model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post dpli (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post NULL dpli model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cfc+dpli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% SMN: MINJURY
final_dataset_SMN = final_dataset_MInjury_SMN;
final_labels_SMN = labels_MInjury_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN MInjury dpli+cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury dpli+cfc (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury NULL dpli+cfc model: ' num2str(mean(accuracy))]);

%% SMN: NEUTRAL - dpli+cfc model
final_dataset_SMN = final_dataset_Neutral_SMN;
final_labels_SMN = labels_Neutral_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN Neutral dpli+cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral dpli+cfc (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral NULL dpli+cfc model: ' num2str(mean(accuracy))]);

%% SMN: RESTINGSTATEPOST - dpli+cfc model
final_dataset_SMN = final_dataset_rsEEG_Post_SMN;
final_labels_SMN = labels_rsEEG_Post_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN rsEEG_Post dpli+cfc model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post dpli+cfc (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post NULL dpli+cfc model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% arr_avgamp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,1001:2000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,1001:2000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,1001:2000);

%% SMN: MINJURY
final_dataset_SMN = final_dataset_MInjury_SMN;
final_labels_SMN = labels_MInjury_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN MInjury arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% SMN: NEUTRAL - arr_avgamp model
final_dataset_SMN = final_dataset_Neutral_SMN;
final_labels_SMN = labels_Neutral_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN Neutral arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% SMN: RESTINGSTATEPOST - arr_avgamp model
final_dataset_SMN = final_dataset_rsEEG_Post_SMN;
final_labels_SMN = labels_rsEEG_Post_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN rsEEG_Post arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% coh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_rest_1001_rsEEG_Post_mRMRiterateResults_SMN_mivo.mat')
final_dataset_rsEEG_Post_SMN = final_dataset_mRMR(:,2001:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_MInjury_mRMRiterateResults_SMN_mivo.mat')
final_dataset_MInjury_SMN = final_dataset_mRMR(:,2001:3000);
load('D:\research_data\HCWMIVO\all_data\ML_x\20210409_HCWMIVO_EEGfMRI-1001\EEG_Features\mrmr\all_but_dpli\Rev_task_1001_Neutral_mRMRiterateResults_SMN_mivo.mat')
final_dataset_Neutral_SMN = final_dataset_mRMR(:,2001:3000);

%% SMN: MINJURY
final_dataset_SMN = final_dataset_MInjury_SMN;
final_labels_SMN = labels_MInjury_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN MInjury arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN MInjury NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% SMN: NEUTRAL - arr_avgamp model
final_dataset_SMN = final_dataset_Neutral_SMN;
final_labels_SMN = labels_Neutral_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN Neutral arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN Neutral NULL arr_avgamp model: ' num2str(mean(accuracy))]);

%% SMN: RESTINGSTATEPOST - arr_avgamp model
final_dataset_SMN = final_dataset_rsEEG_Post_SMN;
final_labels_SMN = labels_rsEEG_Post_SMN;
% Create indices for cross-validation
indices = crossvalind('Kfold', size(final_dataset_SMN, 1), numFolds);
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
disp(['Average Accuracy, SMN rsEEG_Post arr_avgamp model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post arr_avgamp (tested on training data) model: ' num2str(mean(accuracy))]);

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
disp(['Average Accuracy, SMN rsEEG_Post NULL arr_avgamp model: ' num2str(mean(accuracy))]);

