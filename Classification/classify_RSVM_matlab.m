function [TrainAccuracy, TestAccuracy, MdlSVM] = classify_RSVM_matlab(X,Y,kernel,testTrainPartition,varargin)

% kernel can be 'RBF', or 'linear', or 'polynomial'
% returns accuracy, AUCsvm, specificity and sensitivity

%% Create Testing and Training Partitions:
% tabulate(Y); % Gives the distribution of the classes in Y
if testTrainPartition ~= 0
    CVP = cvpartition(Y,'holdout',1-testTrainPartition);
    idxTrain = training(CVP);           % Training-set indices
    idxTest = test(CVP);                % Test-set indices
else
    idxTrain = varargin{1};           % Training-set indices
    idxTest = varargin{2};                % Test-set indices 
end

%% Feature Scaling:
X = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

%% Feature Selection:

%% Train an SVM Model:
% MdlSVM = fitrsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
%     'KernelFunction',kernel,'KFold',5,'KernelScale','auto','FitPosterior',1);
try
    MdlSVM = fitrsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
        'KernelFunction',kernel,'KernelScale','auto','FitPosterior',1);
catch e
    if strcmp(e.message,'FitPosterior is not a valid parameter name.')
        MdlSVM = fitrsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
            'KernelFunction',kernel,'KernelScale','auto');
    else
        MdlSVM = [];
    end
end

% Test the SVM Model:
% [YTrainhat,YTrainhat_NegLoss,YTrainhat_score,YTrainhat_posterior] = resubPredict(MdlSVM);
% [YTesthat,YTesthat_score] = predict(MdlSVM,X(idxTest,:));

YTrainhat = predict(MdlSVM,X(idxTrain,:)); 
YTesthat = predict(MdlSVM,X(idxTest,:));

%% Accumulate results:
% TrainAccuracy = resubLoss(MdlSVM);
% TestAccuracy = kfoldLoss(MdlSVM);
try    
    % TrainAccuracy = loss(MdlSVM,X(idxTrain,:),Y(idxTrain));
    % TestAccuracy = loss(MdlSVM,X(idxTest,:),Y(idxTest));
    
    TrainAccuracy = sqrt(sum((Y(idxTrain)-YTrainhat).^2)/length(Y(idxTrain))); 
    TestAccuracy = sqrt(sum((Y(idxTest)-YTesthat).^2)/length(Y(idxTest)));
    
catch e
    TrainAccuracy = NaN;
    TestAccuracy = NaN;
end
% varargout = []; varargout{1} = perfcurve_stats;
% perfcurve_stats = cell(1,5);
% % [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(MdlSVM.Y,(score(:,2)),1);
% figure; plot(Xsvm, Ysvm); hold on; plot(OPTROCPT(1),OPTROCPT(2),'ro'); xlabel('False positive rate'); ylabel('True positive rate'); title('ROC for Matlab SVM Classification');
% specificity = 1-OPTROCPT(1); % Specificity = 1 - FalsePositiveRate
% sensitivity = OPTROCPT(2); % Sensitivity = TruePositiveRate