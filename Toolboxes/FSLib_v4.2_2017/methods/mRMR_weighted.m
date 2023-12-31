function [fea, score] = mRMR_weighted(X_train, Y_train, weights, K)
% Matlab Code-Library for Feature Selection
% Support: Giorgio Roffo email: giorgio.roffo@univr.it
%  If you use our toolbox please cite our paper:
% 
%  BibTex
%  ------------------------------------------------------------------------
%     @InProceedings{Roffo_2015_ICCV,
%     author = {Roffo, Giorgio and Melzi, Simone and Cristani, Marco},
%     title = {Infinite Feature Selection},
%     journal = {The IEEE International Conference on Computer Vision (ICCV)},
%     month = {June},
%     year = {2015}
%     }
%  ------------------------------------------------------------------------
% MID scheme according to MRMR
%
% By Hanchuan Peng
% April 16, 2003
%
% fprintf('\n+ Feature selection method: mRMR \n');

bdisp=0;

num_discrete_bins = 10;

% Discretize X_train:
if length(unique(X_train)) < num_discrete_bins
    [X_train,X_train_bins] = discretize(X_train,num_discrete_bins);
end

nd = size(X_train,2);
nc = size(X_train,1);

if (size(Y_train,1) ~= nc) && (size(Y_train,2) == nc); Y_train = Y_train'; 
elseif (size(Y_train,1) ~= nc); error('Size of the Label vector does not match the number of observations in Data vector'); end

if (size(weights,1) ~= nc) && (size(weights,2) == nc); weights = weights'; 
elseif (size(weights,1) ~= nc); error('Size of the Weights vector does not match the number of observations in Data vector'); end

t1=cputime;
for i=1:nd
   t(i) = WeightedMIToolboxMex(4,weights, X_train(:,i),Y_train);
   %t(i) = mutualinfo(X_train(:,i), Y_train);
end
% fprintf('calculate the marginal dmi costs %5.1fs.\n', cputime-t1);

[tmp, idxs] = sort(-t);
%fea_base = idxs(1:K);

fea = [];
fea(1) = idxs(1);

%KMAX = min(1000,nd); %500

idxleft = idxs(2:K);

k=1;
% if bdisp==1,
% % fprintf('k=1 cost_time=(N/A) cur_fea=%X_train #left_cand=%X_train\n', ...
% %       fea(k), length(idxleft));
% end;

for k=2:K
    t1=cputime;
    ncand = length(idxleft);
    curlastfea = length(fea);
    
    for i=1:ncand
        t_mi(i) = WeightedMIToolboxMex(4, weights ,X_train(:,idxleft(i)),Y_train);
        %t_mi(i) = mutualinfo(X_train(:,idxleft(i)), Y_train);
        mi_array(idxleft(i),curlastfea) = getmultimi(X_train(:,fea(curlastfea)), X_train(:,idxleft(i)),weights);
        c_mi(i) = mean(mi_array(idxleft(i), :));
    end
    
    [score(k), fea(k)] = max(t_mi(1:ncand) - c_mi(1:ncand));
    
    tmpidx = fea(k); fea(k) = idxleft(tmpidx); idxleft(tmpidx) = [];
    
    %    if bdisp==1,
    % %    fprintf('k=%X_train cost_time=%5.4f cur_fea=%X_train #left_cand=%X_train\n', ...
    %       k, cputime-t1, fea(k), length(idxleft));
    %    end;
end

return

%===================================== 
% function c = getmultimi(da, dt, weights)
% for i=1:size(da,2)
%     c(i) = WeightedMIToolboxMex(4, weights, da(:,i), dt);
%     % c(i) = mutualinfo(da(:,i), dt);
% end
%     
