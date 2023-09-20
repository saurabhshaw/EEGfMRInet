% Compute discrete-time Markov chain:



% First 
X = curr_Features_classify(testIdx_SUB,Features_ranked_mRMR_SUB);
Y = curr_YY_final_classify(testIdx_SUB);

X_scaled = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

[YTesthat, testaccur, YTesthat_posterior] = svmpredict(Y', X_scaled, Model_SUB{2},' -b 1');

%% Look at the ones that are very confident:
picked_IDX = find(sum(YTesthat_posterior > 0.75,2));
TestAccuracy_confident = sum(YTesthat(picked_IDX) == Y(picked_IDX)')/length(picked_IDX)

% Use the confident time points for individualized mRMR:
Featurefiles_directory_ALL
Featurefiles_basename_ALL
curr_subIDX = curr_labels_mRMR_subIDX(find(trial_select_bin));
select_testIDX = (current_test_block == curr_subIDX);
curate_features_mRMR_compiled([Featurefiles_basename '_CLASS' CONN_cfg.class_types], Featurefiles_directory, YY_final, max_features, task_dir, base_path)

%% Compute number of transistions:
curr_sequence = YTesthat(picked_IDX);
Y_labels = unique(Y);
Y_P = zeros(length(Y_labels));
for i = 1:length(Y_labels)    
    for j = 1:length(Y_labels)
        % Compute number of times state i transistions to state j:
        curr_sequence_i = diff(curr_sequence == i) == -1; % switching out of i
        curr_sequence_j = diff(curr_sequence == j) == 1; % switching into j
        Y_P(i,j) = sum(curr_sequence_i & curr_sequence_j);
        
    end
end

%% Compute transistion matrix:
orig_mc = dtmc(Y_P,'StateNames',["CEN" "DMN" "SN"]);
orig_mc.P