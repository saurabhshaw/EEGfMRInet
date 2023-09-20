function [TrainAccuracy, TestAccuracy, MdlSVM, perfcurve_stats] = classify_SVM_matlab(X,Y,kernel,testTrainPartition)

% kernel can be 'RBF', or 'linear', or 'polynomial'
% returns accuracy, AUCsvm, specificity and sensitivity

%% Create Testing and Training Partitions:
% tabulate(Y); % Gives the distribution of the classes in Y
CVP = cvpartition(Y,'holdout',1-testTrainPartition);
idxTrain = training(CVP);           % Training-set indices
idxTest = test(CVP);                % Test-set indices

%% Feature Scaling:
X = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

%% Feature Selection:

%% Train an SVM Model:
try
    MdlSVM = fitcsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
        'KernelFunction',kernel,'KernelScale','auto','FitPosterior',1);
    
    % Test the SVM Model:
    [YTrainhat,YTrainhat_NegLoss,YTrainhat_score,YTrainhat_posterior] = resubPredict(MdlSVM);
    [YTesthat,YTesthat_score] = predict(MdlSVM,X(idxTest,:));

catch e
    if strcmp(e.message,'FitPosterior is not a valid parameter name.')
        MdlSVM_curr = fitcsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
            'KernelFunction',kernel,'KernelScale','auto');
        MdlSVM = fitPosterior(MdlSVM_curr);
        
        % Test the SVM Model:
        [YTrainhat] = resubPredict(MdlSVM);
        [YTesthat,YTesthat_score] = predict(MdlSVM,X(idxTest,:));
    end
end

%% Accumulate results:
TrainAccuracy = sum(YTrainhat == Y(idxTrain))/length(idxTrain);
TestAccuracy = sum(YTesthat == Y(idxTest))/length(idxTest);
[Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(Y(idxTest),(YTesthat_score(:,2)),1);
perfcurve_stats = {Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT};
% % [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(MdlSVM.Y,(score(:,2)),1);
% figure; plot(Xsvm, Ysvm); hold on; plot(OPTROCPT(1),OPTROCPT(2),'ro'); xlabel('False positive rate'); ylabel('True positive rate'); title('ROC for Matlab SVM Classification');
% specificity = 1-OPTROCPT(1); % Specificity = 1 - FalsePositiveRate
% sensitivity = OPTROCPT(2); % Sensitivity = TruePositiveRate