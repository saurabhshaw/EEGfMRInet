
% dataset_name = 'P1001_Pretrain';
% srate = 512;
% feature_names = {'COH', 'PAC', 'PLI', 'dPLI', 'CFC_SI'};
% feature_names = {'dPLI'};
% featureVar_to_load = 'final'; % Can be 'final' or 'full', leave 'final'
% max_features = 1000;%keep this CPU-handle-able
% testTrainSplit = 0.75; %classifier - trained on 25%
% num_CV_folds = 20; %classifier - increased folds = more accurate but more computer-intensive??

% [base_path_rc, base_path_rd] = setPaths();
% base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
% base_path = pwd;

% curr_dir_data = [base_path_rd filesep 'Analyzed_data' filesep 'DatabaseBCI'];
% curr_dir_data = ['W:\Experiments\DatabaseBCI\Analyzed_data'];
% load([curr_dir_data filesep 'Curated_dataset.mat']);

%% Run classification:
Y_unique = unique(Y);
for i = 1:length(Y_unique)
    for j = 1:length(Y_unique)
        if i < j
            Y1 = Y_unique{i}; 
            Y2 = Y_unique{j};
            
            Y1_find = find(cellfun(@(x)strcmp(x,Y1),Y));
            Y2_find = find(cellfun(@(x)strcmp(x,Y2),Y));
            
            YY = [Y1_find Y2_find];
            
            XX = X(:,:,YY);
            YY_final = [zeros(1,length(Y1_find)) ones(1,length(Y2_find))];
            
            %%%%%%%%%%% Classifier code runs here %%%%%%%%%%%
            % Using XX and YY_final
            
            % Setup code for feature computation:
            EEG = []; EEG.data = XX; EEG.srate = srate;            
            curr_dir = [curr_dir_data filesep Y1 'vs' Y2];
            if isempty(dir(curr_dir)) mkdir(curr_dir); end 
            
            % compute_features_attentionbci
            if isempty(dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*Epoch*.mat']))
                fprintf(['***************************** Starting Feature Computation *****************************']);
                tic; compute_features_compiled(EEG,curr_dir,dataset_name,feature_names,base_path); toc
            end
            
            % Curate features:
            fprintf(['***************************** Curating Computed Features *****************************']);
            Featurefiles_directory = [curr_dir filesep 'EEG_Features'];
            Featurefiles_basename = ['Rev_' dataset_name];
            [compute_feat, Features, final_FeatureIDX] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, 0, 1);

            %% Select Features and Classify the data:
            fprintf(['***************************** Starting Feature Selection and Classification *****************************']);

            % Run Feature Selection:
            nclassesIdx = randperm(length(YY_final));
            trial_data_num = YY_final(nclassesIdx);
            [Features_ranked_mRMR, Features_scores_mRMR] = mRMR(Features(nclassesIdx,:),trial_data_num(nclassesIdx)',max_features);
            
            % Classify:
            TrainAccuracy = zeros(1,num_CV_folds); TestAccuracy = zeros(1,num_CV_folds); Model = cell(1,num_CV_folds);
            parfor i = 1:num_CV_folds
                [TrainAccuracy(i), TestAccuracy(i), Model{i}] = classify_SVM_libsvm(Features(nclassesIdx,Features_ranked_mRMR),trial_data_num(nclassesIdx)','RBF',testTrainSplit);
            end
            
            save([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults'],'TrainAccuracy','TestAccuracy','Model','Features_ranked_mRMR','Features_scores_mRMR','final_FeatureIDX');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
    end
end