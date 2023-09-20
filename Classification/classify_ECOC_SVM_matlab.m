function [TrainAccuracy, TestAccuracy, MdlECOC] = classify_ECOC_SVM_matlab(X,Y,kernel,testTrainPartition,varargin)

warning off
% If testTrainPartition is 0 - can specify separate testing and training
% matrices
% kernel can be 'RBF', or 'linear', or 'polynomial'
% returns accuracy, AUCsvm, specificity and sensitivity

%% Create Testing and Training Partitions:
if testTrainPartition ~= 0
    CVP = cvpartition(Y,'holdout',1-testTrainPartition);
    idxTrain = training(CVP);           % Training-set indices
    idxTest = test(CVP);                % Test-set indices
else
    idxTrain = varargin{1};           % Training-set indices
    idxTest = varargin{2};                % Test-set indices 
end

%% Feature Selection:
% Currently empty

%% Train an ECOC Model:
compute_posterior = 0;
optimize = 0;
options = statset('UseParallel',true);
tSVM = templateSVM('KernelFunction',kernel,'Standardize',true,...
    'KernelScale','auto');
% C = templateECOC('Learners',tSVM);
% designecoc(K,name)
if ~optimize    
    MdlECOC = fitcecoc(X(idxTrain,:),Y(idxTrain),'Learners',tSVM,...
        'FitPosterior',compute_posterior,'Options',options);    
else
    MdlECOC = fitcecoc(X(idxTrain,:),Y(idxTrain),'Learners',tSVM,...
        'FitPosterior',compute_posterior,'Options',options,...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus'));
end

%% Test the ECOC Model:
if compute_posterior
    [YTrainhat,YTrainhat_NegLoss,YTrainhat_score,YTrainhat_posterior] = resubPredict(MdlECOC);
    [YTesthat,YTesthat_score,~,YTesthat_posterior] = predict(MdlECOC,X(idxTest,:));
else
    [YTrainhat] = resubPredict(MdlECOC);
    [YTesthat] = predict(MdlECOC,X(idxTest,:));
end
%% Accumulate results:
% cp = classperf
TrainAccuracy = sum(YTrainhat == Y(idxTrain))/length(idxTrain);
TestAccuracy = sum(YTesthat == Y(idxTest))/length(idxTest);
% [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(Y(idxTest),(Yhat_score(:,2)),1);
% [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(MdlSVM.Y,(score(:,2)),1);
% figure; plot(Xsvm, Ysvm); hold on; plot(OPTROCPT(1),OPTROCPT(2),'ro'); xlabel('False positive rate'); ylabel('True positive rate'); title('ROC for Matlab SVM Classification');
% specificity = 1-Xsvm(OPTROCPT(1)); % Specificity = 1 - FalsePositiveRate
% sensitivity = Ysvm(OPTROCPT(2)); % Sensitivity = TruePositiveRate