function [Yhat, varargout] = predict_SVM_libsvm(X,Model,varargin)                

% Set compute_posterior flag:
if ~isempty(varargin)
    compute_posterior = varargin{1};
else
    compute_posterior = 0;
end

% Create predict string:
predict_control_string = '';
if compute_posterior predict_control_string = [predict_control_string, ' -b 1']; end

% Scale the input Features between 0 and 1 (Matches classify_SVM_libSVM):
X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

% Predict:
[Yhat, ~, Yhat_posterior] = svmpredict(zeros(size(X_scaled,1),1), X_scaled, Model, predict_control_string);

% Order the posterior probabilities according to the label order:
if compute_posterior 
    varargout = [];
    [~,sortIdx] = sort(Model.Label);
    varargout{1} = Yhat_posterior(:,sortIdx);
end
