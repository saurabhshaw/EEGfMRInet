function [output_features, output_scores,dataset_mRMR,feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,dataset_mRMR,feature_labels_mRMR,varargin)
% These are the lines that are run at each level of the loop to get the multi-level mRMR output:

%% Run mRMR:
% using fslib mrmr
[output_features, output_scores] = mRMR(curr_dataset_mRMR,curr_labels_mRMR,min(max_features,size(curr_dataset_mRMR,2)));
% using matlab mrmr
[output_features_m, output_scores_m] = fscmrmr(curr_dataset_mRMR,curr_labels_mRMR);output_features_m=output_features_m(1:max_features);

%% Isolate mRMR dataset for next level run:
dataset_mRMR = cat(2,dataset_mRMR,curr_dataset_mRMR(:,output_features));

%% Create the feature labels for the identified features:

if isempty(varargin) % Will be empty for the inner-most loop
    prev_feature_labels_mRMR = cell(1,length(output_features));
    prev_feature_labels_mRMR = cellfun(@(x)'',prev_feature_labels_mRMR,'un',0);
    
    % Find the matrix indices:
    curr_feature_labels_mRMR = zeros(length(curr_feature_size),length(output_features));
    code_to_eval = '[curr_feature_labels_mRMR(';
    for i = 1:length(curr_feature_size)
        code_to_eval = [code_to_eval num2str(i) ',:)'];
        if i ~= length(curr_feature_size)
            code_to_eval = [code_to_eval ',curr_feature_labels_mRMR('];
        end
    end
    code_to_eval = [code_to_eval '] = ind2sub(curr_feature_size,output_features);']; %example: '[curr_feature_labels_mRMR(1,:),curr_feature_labels_mRMR(2,:)] = ind2sub(curr_feature_size,output_features);'
    eval(code_to_eval);
    
else % This is one of the outer loops
    
    master_feature_labels_mRMR = varargin{1};    
    prev_feature_labels_mRMR = master_feature_labels_mRMR(output_features);
    if length(varargin) == 1 % For single frequency runs where the feature vectors are always full
        curr_feature_labels_mRMR = floor(output_features/curr_feature_size(2)) + 1;
    elseif length(varargin) == 2 % For cross-frequency runs where the feature vectors might be missing
        curr_feature_labels_mRMR = varargin{2}(output_features);
    end
end



% Convert the indices to string:
curr_feature_labels_mRMR_string = cell(1,size(curr_feature_labels_mRMR,2));
for i = 1:size(curr_feature_labels_mRMR,2)
    temp_string = prev_feature_labels_mRMR{i};
    if ~isempty(temp_string) temp_string = ['_' temp_string]; end
    
    for j = size(curr_feature_labels_mRMR,1):-1:1 % Start from the end and move towards the beginning
        % temp_string = [temp_string num2str(curr_feature_labels_mRMR(j,i))];
        
        temp_string = [num2str(curr_feature_labels_mRMR(j,i)) temp_string];
        if j~= 1 temp_string = ['_' temp_string]; end
    end
    curr_feature_labels_mRMR_string{i} = temp_string;
end

feature_labels_mRMR = cat(2,feature_labels_mRMR,curr_feature_labels_mRMR_string);

%% Old code - this is what this function replaces:
%                 [elec_output_features{n}, elec_output_scores{n}] = mRMR(curr_dataset_mRMR,curr_labels_mRMR,min(max_features,size(curr_dataset_mRMR,2)));
%                 
%                 elec_dataset_mRMR = cat(2,elec_dataset_mRMR,curr_dataset_mRMR(:,elec_output_features{n}));
%                 elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR,curr_feature_labels_mRMR(elec_output_features{n}));
%                 
