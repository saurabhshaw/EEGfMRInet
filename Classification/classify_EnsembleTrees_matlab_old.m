function [TrainAccuracy, TestAccuracy, C, perfcurve_stats, predImp] = classify_EnsembleTrees_matlab(X,Y,numTrees,testTrainPartition,regression)

% Create Testing and Training Partitions:
% tabulate(Y); % Gives the distribution of the classes in Y
CVP = cvpartition(Y,'holdout',1-testTrainPartition);
idxTrain = training(CVP);           % Training-set indices
idxTest = test(CVP);                % Test-set indices

% use surrogate splits since there are missing values in the data
% tTree = templateTree('surrogate','on');
% tEnsemble = templateEnsemble('GentleBoost',100,tTree);

% numTrees = 100;
tTree = templateTree('surrogate','on');
% tEnsemble = templateEnsemble('GentleBoost',numTrees,tTree);
% C = fitensemble(X,Y,'AdaBoostM1',nTrees,'Tree');

if length(unique(Y)) == 2
    boostmethod = 'AdaBoostM1';
elseif regression
    boostmethod = 'LSBoost';
else
    boostmethod = 'AdaBoostM2';
end

C = fitensemble(X(idxTrain,:),Y(idxTrain),boostmethod,numTrees,tTree); % Standardize
[YTrainhat] = resubPredict(C);
% [YTesthat,YTesthat_score] = predict(C,X(idxTest,:));
[YTesthat] = predict(C,X(idxTest,:));

TrainAccuracy = sum(YTrainhat == Y(idxTrain))/length(idxTrain);
TestAccuracy = sum(YTesthat == Y(idxTest))/length(idxTest);
% [Xtree,Ytree,Ttree,AUCtree,OPTROCPT] = perfcurve(Y(idxTest),(YTesthat_score(:,2)),1);
% perfcurve_stats = {Xtree,Ytree,Ttree,AUCtree,OPTROCPT};
perfcurve_stats = cell(1,5);
predImp = predictorImportance(C);

% rsLoss = resubLoss(C,'Mode','Cumulative');
rsLoss = resubLoss(C,'Mode','Cumulative');

%% Plot
figure;
bar(predImp);
h = gca;
h.XTick = 1:2:h.XLim(2);
title('Predictor Importances');
xlabel('Predictor');
ylabel('Importance measure');

[~,idxSort] = sort(predImp,'descend');
idx5 = idxSort(1:5);

% plot(rsLoss);
% xlabel('Number of Learning Cycles');
% ylabel('Resubstitution Loss');