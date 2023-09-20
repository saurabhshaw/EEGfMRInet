curr_model = [];
curr_model.model_features = mRMRiterateResults.currFeatures_curated;

num_feats_per_type = 1000;
features_IDX = [];

for i = 1:length(mRMRiterateResults.currFeatures_curated) features_IDX = [features_IDX ones(1,num_feats_per_type)*i]; end
curr_model.final_feature_labels = cellfun(@(x,y) [num2str(y) '_' x],mRMRiterateResults.final_feature_labels_mRMR,arrayfun(@(x) x,features_IDX,'un',0),'un',0);

tic; [Features_test] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile); toc;