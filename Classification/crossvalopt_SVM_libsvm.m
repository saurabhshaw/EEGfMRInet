% function [bestc, bestg, varargout] = crossvalopt_SVM_libsvm(X, Y, optimize_kernel)
% Use this to find the optimal hyperparameters for running SVM
% Optimizes the following:
% kernel = linear/RBF/polynomial/sigmoid
% C = misclassification penalty/box constraint
% gamma = kernel scale/kernel bandwidth
%

range_log2c = -1:3;
range_log2g = -4:1;
if optimize_kernel
    range_kernel = 0:3;
end

num_cvfolds = 5;

if optimize_kernel
    cv_vect = zeros(length(range_kernel),length(range_log2c),length(range_log2g));
    for idx_ker = 1:length(range_kernel)
        for idx_c = 1:length(range_log2c)
            log2c = range_log2c(idx_c);
            parfor idx_g = 1:length(range_log2g)
                log2g = range_log2g(idx_g);
                cmd = ['-v ' num2str(num_cvfolds) ' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
                cv_vect(idx_ker, idx_c, idx_g) = svmtrain(Y, X, cmd);
                
                fprintf('%%%%%%%%%%%%%%%%%%%% Kernel %g; Log2C Value %g; Log2g Value %g; Accuracy %g %%%%%%%%%%%%%%%%%%%%\n', range_kernel(idx_ker), log2c, log2g, cv_vect(idx_ker, idx_c, idx_g));
            end
        end
    end
    
    [~,I] = max(cv_vect(:));
    [bestker_idx, bestc_idx, bestg_idx] = ind2sub(size(cv_vect),I);
    
%     [cv_vect_maxc,bestc_idx] = max(cv_vect,[],1); cv_vect_maxc = squeeze(cv_vect_maxc); bestc_idx = squeeze(bestc_idx);
%     [cv_vect_maxc_maxg,bestg_idx] = max(cv_vect_maxc,[],1); bestc_idx = bestc_idx(bestg_idx);
%     [cv_vect_maxc_maxg_maxker,bestker_idx] = max(cv_vect_maxc_maxg); bestc_idx = bestc_idx(bestker_idx); bestg_idx = bestg_idx(bestker_idx);
%     
    bestc = 2^range_log2c(bestc_idx); bestg = 2^range_log2g(bestg_idx); bestker = range_kernel(bestker_idx);
    varargout{1} = bestker;
    
else
    
    cv_vect = zeros(length(range_log2c),length(range_log2g));
    for idx_c = 1:length(range_log2c)
        log2c = range_log2c(idx_c);
        parfor idx_g = 1:length(range_log2g)
            log2g = range_log2g(idx_g);
            cmd = ['-v ' num2str(num_cvfolds) ' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv_vect(idx_c, idx_g) = svmtrain(Y, X, cmd);
            
            fprintf('%%%%%%%%%%%%%%%%%%%% Log2C Value %g; Log2g Value %g; Accuracy %g %%%%%%%%%%%%%%%%%%%%\n', log2c, log2g, cv_vect(idx_c, idx_g));
        end
    end
    
    
    [~,I] = max(cv_vect(:));
    [bestc_idx, bestg_idx] = ind2sub(size(cv_vect),I);
    
%     [cv_vect_maxc,bestc_idx] = max(cv_vect,[],1);
%     [cv_vect_maxc_maxg,bestg_idx] = max(cv_vect_maxc); bestc_idx = bestc_idx(bestg_idx);
    bestc = 2^range_log2c(bestc_idx); bestg = 2^range_log2g(bestg_idx);
    
end
% if (cv >= bestcv)
%     bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
% end
% fprintf('%%%%%%%%%% best c=%g, g=%g, rate=%g) %%%%%%%%%%\n', bestc, bestg, bestcv);