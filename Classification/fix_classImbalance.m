function [trial_select_bin,varargout] = fix_classImbalance(YY_final,methodlogy,varargin)

% Fix class imbalance:
trial_data = YY_final;         
trial_data_unique = unique(trial_data); % Convert labels to numbers

switch methodlogy
    case 'balance'
        % Select the trials corresponding to the classes specified above:
        % [~,trial_data_num] = ismember(trial_data,trial_data_unique); % Convert labels to numbers
        % nclassesIdx = []; for i = 1:length(nclasses) nclassesIdx = [nclassesIdx find(trial_data_num == nclasses(i))]; end % Pick the trials corresponding to the relevant classes
        % Find number of trials of each trial type
        nclassesIdx = cell(1,length(trial_data_unique)); for i = 1:length(trial_data_unique) nclassesIdx{i} = find(trial_data == trial_data_unique(i)); end
        nclassesSize = cellfun(@(x) length(x),nclassesIdx);
        [~,class_to_fix] = max(nclassesSize);
        target_size = ceil(mean(nclassesSize(setdiff([1:length(nclassesSize)],class_to_fix))));
        nclassesIdx{class_to_fix} = nclassesIdx{class_to_fix}(randperm(length(nclassesIdx{class_to_fix}),target_size));
        nclassesIdx = cat(2,nclassesIdx{:});
        
        % Get class weights:
        nclassesPercentage = nclassesSize./sum(nclassesSize);
        [~,min_classes] = min(nclassesPercentage);
        nclassesPercentage_minNorm = nclassesPercentage./min(nclassesPercentage);
        class_weights = 1./nclassesPercentage_minNorm;
        varargout{1} = class_weights;

    case 'remove'
        class_to_remove = varargin{1};
        final_trial_data_unique = setdiff(trial_data_unique,class_to_remove);
        nclassesIdx = []; for i = 1:length(final_trial_data_unique) nclassesIdx = [nclassesIdx find(trial_data == final_trial_data_unique(i))]; end % Pick the trials corresponding to the relevant classes

        
    case 'weights'
        nclassesIdx = cell(1,length(trial_data_unique)); for i = 1:length(trial_data_unique) nclassesIdx{i} = find(trial_data == trial_data_unique(i)); end
        nclassesSize = cellfun(@(x) length(x),nclassesIdx);
        nclassesPercentage = nclassesSize./sum(nclassesSize);
        [~,min_classes] = min(nclassesPercentage);
        nclassesPercentage_minNorm = nclassesPercentage./min(nclassesPercentage);
        class_weights = 1./nclassesPercentage_minNorm;
        varargout{1} = class_weights;
        nclassesIdx = 1:length(trial_data);
end

trial_select_bin = zeros(size(trial_data)); trial_select_bin(nclassesIdx) = 1;
                        
%% old code:
%                         % Fix class imbalance:
%                         % Select the trials corresponding to the classes specified above:
%                         trial_data = YY_final; trial_data_unique = unique(trial_data); % Convert labels to numbers
%                         % [~,trial_data_num] = ismember(trial_data,trial_data_unique); % Convert labels to numbers
%                         % nclassesIdx = []; for i = 1:length(nclasses) nclassesIdx = [nclassesIdx find(trial_data_num == nclasses(i))]; end % Pick the trials corresponding to the relevant classes
%                         % Find number of trials of each trial type
%                         nclassesIdx = cell(1,length(trial_data_unique)); for i = 1:length(trial_data_unique) nclassesIdx{i} = find(trial_data == trial_data_unique(i)); end
%                         nclassesSize = cellfun(@(x) length(x),nclassesIdx);
%                         [~,class_to_fix] = max(nclassesSize); 
%                         target_size = ceil(mean(nclassesSize(setdiff([1:length(nclassesSize)],class_to_fix))));
%                         nclassesIdx{class_to_fix} = nclassesIdx{class_to_fix}(randperm(length(nclassesIdx{class_to_fix}),target_size));
%                         nclassesIdx = cat(2,nclassesIdx{:});
%                         trial_select_bin = zeros(size(trial_data)); trial_select_bin(nclassesIdx) = 1;
%                         
