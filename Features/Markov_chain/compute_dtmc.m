% Compute discrete-time Markov chain:

function [orig_mc,Y_P] = compute_dtmc(curr_sequence,num_states)

% % First 
% X = curr_Features_classify(testIdx_SUB,Features_ranked_mRMR_SUB);
% Y = curr_YY_final_classify(testIdx_SUB);
% 
% X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));
% 
% [YTesthat, testaccur, YTesthat_posterior] = svmpredict(Y', X_scaled, Model_SUB{2},' -b 1');
% 
% %% Isolate the time points that were confident:
% picked_IDX = find(sum(YTesthat_posterior > 0.75,2));
% TestAccuracy_confident = sum(YTesthat(picked_IDX) == Y(picked_IDX)')/length(picked_IDX)
% 
% %% Use the confident time points for individualized mRMR:
% Featurefiles_directory_ALL
% Featurefiles_basename_ALL
% 
% %curr_subIDX = curr_labels_mRMR_subIDX(find(trial_select_bin));
% %select_testIDX = (current_test_block == curr_subIDX);
% 
% [~,curr_labels_mRMR_subIDX2curr_subIDX] = find(trial_select_bin);
% origSpace_select_testIDX = curr_labels_mRMR_subIDX2curr_subIDX(select_testIDX);
% optimal_testIDX = origSpace_select_testIDX(picked_IDX);
% % curate_features_mRMR_compiled([Featurefiles_basename '_CLASS' CONN_cfg.class_types], Featurefiles_directory, YY_final, max_features, task_dir, base_path)
% 
% % Give optimal_testIDX as curr_dataset_mRMR_IDX to run mRMR on this custom
% % dataset

%% Compute number of transistions:
% curr_sequence = YTesthat(picked_IDX);
Y_labels = sort(unique(curr_sequence));
%Y_P = zeros(length(Y_labels));
% for i = 1:length(Y_labels)    
%     for j = 1:length(Y_labels)
% Direction of transistion is row_state ---> column_state

Y_P = zeros(num_states);
for i = 1:num_states   
    for j = 1:num_states
        % Compute number of times state i transistions to state j:
        if i ~= j
            curr_sequence_i = diff(curr_sequence == i) == -1; % switching out of i
            curr_sequence_j = diff(curr_sequence == j) == 1; % switching into j
            Y_P(i,j) = sum(curr_sequence_i & curr_sequence_j);
            % if (Y_P(i,j) == 0) Y_P(i,j) = nan; end
            
        else % Self transistion:
            curr_sequence_ij = (curr_sequence == i);
            curr_sequence_i = find(diff(curr_sequence == i) == -1); % switching out of i
            curr_sequence_j = find(diff(curr_sequence == j) == 1); % switching into j
            curr_sequence_ij(curr_sequence_j + 1) = 0;

            Y_P(i,j) = sum(curr_sequence_ij);
            
            %if (Y_P(i,j) == 0) Y_P(i,j) = nan; end
        end
        
    end
end
% Y_P(Y_P==0) = nan;

% Set the probability of the missing states to itself = 1 (needed for dtmc calculation)
missing_states = setdiff([1:num_states],Y_labels);
Y_P(missing_states,missing_states) = 1; 

% Set the probability of transistion of states that are present, but do not
% transistion out of:
still_empty = find(sum(Y_P == 0,2) == num_states);
if ~isempty(still_empty) Y_P(still_empty,still_empty) = 1; end
% if isempty(curr_sequence_i) && isempty(curr_sequence_j) Y_P(i,j) = 1; end % This is not probability matrix - check it

% Convert to probabilities:
% Y_P = Y_P./repmat(sum(Y_P,2),[1,3]);
Y_P = Y_P./repmat(sum(Y_P,2),[1,num_states]); % Changed this to deal with situations where one class is not present in sequence

%% Compute transistion matrix:
orig_mc = dtmc(Y_P,'StateNames',["CEN" "DMN" "SN"]);
%orig_mc.P