function [final_Feature] = get_selected_features(curr_model, Featurefiles_basename, Featurefiles_directory, run_matfile)

currFeatures_curated = curr_model.model_features;
all_Feature_labels = cellfun(@(x)strsplit(x,'_'),curr_model.final_feature_labels,'un',0);

final_Feature = []; % final_FeatureIDX = [];
for i = 1:length(currFeatures_curated)
    % feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
    if run_matfile 
        feat_file = matfile([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
    else
        feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
    end
    curr_feat_size = feat_file.Feature_size;
    
    % Find the features corresponding to the current feature:
    Feature_labels_IDX = cellfun(@(x) str2num(x{1}),all_Feature_labels) == i;
    Feature_labels = cellfun(@(x)x(2:end),all_Feature_labels(Feature_labels_IDX),'un',0);

    curr_window = []; curr_freqBand_A = []; curr_freqBand_B = [];  
    curr_chan_A = []; curr_chan_B = [];

    % Get the subscript indices:
    for j = 1:length(Feature_labels)        
        curr_window = [curr_window str2num(Feature_labels{j}{1})];
        curr_freqBand_A = [curr_freqBand_A str2num(Feature_labels{j}{2})];
        curr_chan_A = [curr_chan_A str2num(Feature_labels{j}{end-1})];
        curr_chan_B = [curr_chan_B str2num(Feature_labels{j}{end})];
        
        if length(Feature_labels{j}) > 4
            curr_freqBand_B = [curr_freqBand_B str2num(Feature_labels{i}{3})];
        else
            curr_freqBand_B = [curr_freqBand_B NaN];
        end
    end
    
    % Get corresponding linear indices of the subscripts:
    if length(curr_feat_size) > 4
        curr_feat_IDX = sub2ind(curr_feat_size,curr_window,curr_freqBand_A,curr_freqBand_B,curr_chan_A,curr_chan_B);
    else        
        curr_feat_IDX = sub2ind(curr_feat_size,curr_window,curr_freqBand_A,curr_chan_A,curr_chan_B);
    end

    if run_matfile
        curr_feat = [];
        allDims = size(feat_file,'Feature'); 
        for j = 1:allDims(2)
            curr_Epoch = feat_file.Feature(1,j);
            curr_feat = cat(1,curr_feat,curr_Epoch{1}(curr_feat_IDX)');
            disp(['Running Feature ' currFeatures_curated{i} ' Epoch ' num2str(j) '/' num2str(allDims(2))]);
        end
    else
        if isfield(feat_file,'analyzedData') 
            curr_feat = cell2mat(feat_file.analyzedData)';
        else
            curr_feat = cell2mat(cellfun(@(x)x(curr_feat_IDX),feat_file.Feature,'un',0))'; 
        end
        % curr_feat = curr_feat(:,curr_feat_IDX);
    end
    
    final_Feature = [final_Feature curr_feat];
end