function [TrainAccuracy, TestAccuracy, Model] = classify_SVM_libsvm(X,Y,kernel,testTrainPartition,varargin)

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

%% Feature Scaling:
X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

% Remove NaN features:
X_isnan = sum(isnan(X_scaled)) > 0; X = X_scaled(:,~X_isnan);
%% Feature Selection:
% Currently empty

%% Train the Multiclass Model:
compute_posterior = 1;
optimize = 0; optimize_kernel = 0;
train_control_string = '-s 0';
predict_control_string = '';
if compute_posterior 
    train_control_string = [train_control_string, ' -b 1'];
    predict_control_string = [predict_control_string, ' -b 1'];
end
if ~optimize_kernel
    switch kernel
        case 'linear'
            train_control_string = [train_control_string, ' -t 0'];
            
        case 'polynomial'
            train_control_string = [train_control_string, ' -t 1'];
            
        case 'RBF'
            train_control_string = [train_control_string, ' -t 2'];
            
        case 'sigmoid'
            train_control_string = [train_control_string, ' -t 3'];
    end
end

% Optimize SVM hyperparameters:
if optimize
    if optmize_kernel
        [bestc, bestg, bestker] = crossvalopt_SVM_libsvm(X, Y);
        train_control_string = [train_control_string ' -t ' num2str(bestker) ' -c ', num2str(bestc), ' -g ', num2str(bestg)];
    else
        [bestc, bestg] = crossvalopt_SVM_libsvm(X, Y);
        train_control_string = [train_control_string ' -c ', num2str(bestc), ' -g ', num2str(bestg)];
    end    
end

% Train and Test Model:
Model = svmtrain(Y(idxTrain), X(idxTrain,:),train_control_string);
[YTrainhat, trainaccur, YTrainhat_posterior] = svmpredict(Y(idxTrain), X(idxTrain,:), Model,predict_control_string);
[YTesthat, testaccur, YTesthat_posterior] = svmpredict(Y(idxTest), X(idxTest,:), Model,predict_control_string);

%% Accumulate results:
TrainAccuracy = sum(YTrainhat == Y(idxTrain))/length(Y(idxTrain));
TestAccuracy = sum(YTesthat == Y(idxTest))/length(Y(idxTest));
% [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(Y(idxTest),(Yhat_score(:,2)),1);
% [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(MdlSVM.Y,(score(:,2)),1);
% figure; plot(Xsvm, Ysvm); hold on; plot(OPTROCPT(1),OPTROCPT(2),'ro'); xlabel('False positive rate'); ylabel('True positive rate'); title('ROC for Matlab SVM Classification');
% specificity = 1-Xsvm(OPTROCPT(1)); % Specificity = 1 - FalsePositiveRate
% sensitivity = Ysvm(OPTROCPT(2)); % Sensitivity = TruePositiveRate

%% Options for svmtrain:
% options:
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)



%%  Need to check these out:
% To plot AUC:
% AUC = plotroc(Y(idxTrain),X(idxTrain,:),model);

% To do binary crossvalidation:
% do_binary_cross_validation(training_label_vector, training_instance_matrix, 'libsvm_options', n_fold);
% [predicted_label, evaluation_result, decision_values] = do_binary_predict(testing_label_vector, testing_instance_matrix, model);

% To do one-vs-rest multiclass SVM modelling:
% model = classify_ovrtrain_libsvm(trainY, trainX, '-c 8 -g 4');
% [pred ac decv] = classify_ovrpredict_libsvm(testY, testX, model);
% fprintf('Accuracy = %g%%\n', ac * 100);
% % Conduct CV on a grid of parameters
% bestcv = 0;
% for log2c = -1:2:3,
%   for log2g = -4:2:1,
%     cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%     cv = get_cv_ac(trainY, trainX, cmd, 3);
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end

% Output of svmpredict:
% Returns:
%   predicted_label: SVM prediction output vector.
%   accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.
%   prob_estimates: If selected, probability estimate vector.