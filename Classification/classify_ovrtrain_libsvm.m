function [model] = classify_ovrtrain_libsvm(y, x, cmd)
% Use this to train a one-vs-rest Multiclass SVM Model
labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    models{i} = svmtrain(double(y == labelSet(i)), x, cmd);
end

model = struct('models', {models}, 'labelSet', labelSet);
