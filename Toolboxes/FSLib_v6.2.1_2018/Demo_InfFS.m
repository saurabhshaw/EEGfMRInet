%**********************************************************************
%**********************************************************************
%  HOW TO CORRECTLY USE THE INFINITE FEATURE SELECTION: INF-FS
%**********************************************************************
%**********************************************************************
% If you use our toolbox (or method included in it), please consider to cite: 
% [1] Roffo, G., Melzi, S., Castellani, U. and Vinciarelli, A., 2017. Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach. arXiv preprint arXiv:1707.07538. 
% [2] Roffo, G., Melzi, S. and Cristani, M., 2015. Infinite feature selection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 4202-4210). 
% [3] Roffo, G. and Melzi, S., 2017, July. Ranking to learn: Feature ranking and selection via eigenvector centrality. In New Frontiers in Mining Complex Patterns: 5th International Workshop, NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016, Revised Selected Papers (Vol. 10312, p. 19). Springer.
% [4] Roffo, G., 2017. Ranking to Learn and Learning to Rank: On the Role of Ranking in Pattern Recognition Applications. arXiv preprint arXiv:1706.05933.
% Demo file for the Feature Selection Library.
% Written by Giorgio Roffo (2018)

% The following code can strongly increase your classification performance.

% This Demo is based on: LIBLINEAR -- A Library for Large Linear Classification
% Free Download: https://www.csie.ntu.edu.tw/~cjlin/liblinear/



% Load your dataset and ground truth here...

% Split the datasets in X_train , Y_train, X_test, and Y_test
% Let's suppose your training data is now loaded in X_train , Y_train


% set a particular small amount of features to find good parameters
fixed_feats = 20;

% Speed up the cross-validation
filename = ['temp_',num2str(randi(10000,1)),'.mat'];
flag=0;

% Cross-Validation for the Infinite Feature Selection
bestcv = 0;
for alpha = [0.5 0.1 0.2 0.3 0.7 0.8 0.9 1 0] % Different alpha parameters
    [ranking, ~] = infFS_fast( X_train , Y_train, alpha , 0 ,flag, 0 , filename);
    flag=1;
    X_train_ = sparse(double(X_train(:,ranking(1:fixed_feats))));
    cmd = ['-q -v 5 -s ',num2str(0),' -c ', num2str(1)];
    cv = train(Y_train, X_train_, cmd);
    if (cv > bestcv)
        bestcv = cv;
        best_a = alpha;
    end
end

% Rank the features on the training data
[ranking, w] = infFS_fast( X_train , Y_train, best_a , 0 ,flag, 0,filename );

% delete the temp file
delete(filename);

% Once we found good mixing coefficients we can train the linear SVM.

% Extract the subset
X_train_ = sparse(double(X_train(:,ranking(1:numFeat))));
X_test_ = sparse(double((X_test(:,ranking(1:numFeat)))));

% Cross-Validation - SVMs parameters
bestcv = 0;
% Extract the subset
for solver=[0 1 2 3] % cross-validate the type of solver
    for log2c = linspace(-11,9,100)
        cmd = ['-q -v 5 -s ',num2str(solver),' -c ', num2str(2^log2c)];
        cv = train(Y_train, X_train_, cmd);
        if (cv > bestcv)
            bestcv = cv; bestc = 2^log2c; bestSolver = solver;
        end
    end
end

% Training 
cmd = ['-q -s ',num2str(bestSolver),' -c ', num2str(bestc)];
model = train(Y_train, X_train_, cmd);

% Testing
[pred , ~, ~] = predict(Y_test,  X_test_, model ,'-q');


accuracy = sum(pred==Y_test)/length(Y_test);

fprintf('Accuracy %.2f \n',accuracy);

disp('Thanks!');



