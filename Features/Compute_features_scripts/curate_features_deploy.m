function [compute_feat, varargout] = curate_features_deploy(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, cell_output, provide_final)
% Features_to_process is a cell containing a string of all of the feature names to curate
% if mRMR_features = 0, then do not run 

% Determine if curate full feature set for each feature, or the final
% computed feature for each feature:
Features_to_process = [];
switch featureVar_to_load
    case 'final'
        if sum(cell2mat(cellfun(@(x) strcmp(x,'BPow'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'BandPowers'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'FBCSP'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'FBCSP'; end

        if sum(cell2mat(cellfun(@(x) strcmp(x,'COH'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'avg_coherence'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PAC'),feature_names,'UniformOutput',0))) compute_feat.compute_PAC = 1; Features_to_process{end+1} = 'arr_avgamp'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PLI'),feature_names,'UniformOutput',0))) compute_feat.compute_PLI = 1; Features_to_process{end+1} = 'PLI_z_score'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'dPLI'),feature_names,'UniformOutput',0))) compute_feat.compute_dPLI = 1; Features_to_process{end+1} = 'dPLI_z_score'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'STE'),feature_names,'UniformOutput',0))) compute_feat.compute_STE = 1; Features_to_process{end+1} = 'STE'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'Bspec'),feature_names,'UniformOutput',0))) compute_feat.compute_Bspec = 1; Features_to_process{end+1} = 'Bspec_features'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'CFC_SI'),feature_names,'UniformOutput',0))) compute_feat.compute_CFC_SI = 1; Features_to_process{end+1} = 'CFC_SI_mag'; end
        
    case 'full'
        if sum(cell2mat(cellfun(@(x) strcmp(x,'BPow'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'avg_coherence'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'FBCSP'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'FBCSP'; end

        if sum(cell2mat(cellfun(@(x) strcmp(x,'COH'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'avg_coherence'; Features_to_process{end+1} = 'all_coherence'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PAC'),feature_names,'UniformOutput',0))) compute_feat.compute_PAC = 1; Features_to_process{end+1} = 'arr_sortamp'; Features_to_process{end+1} = 'arr_plotamp'; Features_to_process{end+1} = 'arr_avgamp'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PLI'),feature_names,'UniformOutput',0))) compute_feat.compute_PLI = 1; Features_to_process{end+1} = 'PLI_z_score'; Features_to_process{end+1} = 'PLI'; Features_to_process{end+1} = 'PLIcorr'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'dPLI'),feature_names,'UniformOutput',0))) compute_feat.compute_dPLI = 1; Features_to_process{end+1} = 'dPLI_z_score'; Features_to_process{end+1} = 'dPLI'; Features_to_process{end+1} = 'dPLIcorr'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'STE'),feature_names,'UniformOutput',0))) compute_feat.compute_STE = 1; Features_to_process{end+1} = 'STE'; Features_to_process{end+1} = 'NSTE'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'Bspec'),feature_names,'UniformOutput',0))) compute_feat.compute_Bspec = 1; Features_to_process{end+1} = 'Bspec_features'; Features_to_process{end+1} = 'Bspec_features_raw_cell'; Features_to_process{end+1} = 'Bspec_features_real_cell'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'CFC_SI'),feature_names,'UniformOutput',0))) compute_feat.compute_CFC_SI = 1; Features_to_process{end+1} = 'CFC_SI_mag'; Features_to_process{end+1} = 'CFC_SI'; Features_to_process{end+1} = 'CFC_SI_theta'; end
end


%% Curate features from all available files:
features_curated = false;
Featurefiles_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_Epoch*.mat']);
Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);
Features_to_process = setdiff(Features_to_process,currFeatures_curated);
% if ~isempty(Featurefiles_dir) && isempty(Featurefiles_curated_dir)
if ~isempty(Featurefiles_dir) && ~isempty(Features_to_process)
    Featurefiles_names = {Featurefiles_dir(:).name}; curr_file_order = cell2mat(cellfun(@(x) str2num(x((strfind(x,'h') + 1):(strfind(x,'.') - 1))),Featurefiles_names,'UniformOutput',0));
    [~, curr_file_order_sorted_idx] = sort(curr_file_order);
    for i = 1:length(Features_to_process) % This is the PARFOR 
        Feature = []; Feature_labels = [];
        for j = 1:length(curr_file_order_sorted_idx)
            
            curr_Epoch = curr_file_order_sorted_idx(j);
            disp(['Running Feature ' Features_to_process{i} ' Epoch ' num2str(curr_Epoch)]);
            curr_file = load([Featurefiles_directory filesep Featurefiles_dir(curr_Epoch).name]);
            
            % Compute derivative features from original features if not included in saved file:
            if ~isfield(curr_file,Features_to_process{i})
                if strcmp(Features_to_process{i},'CFC_SI_mag') || strcmp(Features_to_process{i},'CFC_SI_theta')
                    for m = 1:length(curr_file.CFC_SI{1}) % Number of Windows
                        for n = 1:size(curr_file.CFC_SI{1}{1},2) % Number of Frequency Windows - columns
                            for p = 1:size(curr_file.CFC_SI{1}{1},1) % Number of Frequency Windows 2 - rows
                                if ~isempty(curr_file.CFC_SI{1}{m}{p,n})
                                    num_chan = size(curr_file.CFC_SI{1}{m}{p,n},1);
                                    for ch1 = 1:num_chan                                        
                                        curr_file.CFC_SI_mag{1}{m}{p,n}(ch1,:) = abs(curr_file.CFC_SI{1}{m}{p,n}(ch1,:));
                                        curr_file.CFC_SI_theta{1}{m}{p,n}(ch1,:) = angle(curr_file.CFC_SI{1}{m}{p,n}(ch1,:));
                                    end
                                elseif p == size(curr_file.CFC_SI{1}{1},1)
                                    curr_file.CFC_SI_mag{1}{m}{p,n} = [];
                                end
                            end
                        end
                    end
                end
            end

            
            if isfield(curr_file,'analyzedData')
                curr_file_cell = struct2cell(curr_file.analyzedData); curr_file_cell_var = fields(curr_file.analyzedData);
            else
                curr_file_cell = struct2cell(curr_file); curr_file_cell_var = fields(curr_file);
            end
            curr_feature_idx = cell2mat(cellfun(@(x) strcmp(x,Features_to_process{i}), curr_file_cell_var,'UniformOutput',0));
            curr_Feature = curr_file_cell{curr_feature_idx};
            
            % Give feature output as cell or as a matrix:
            if cell_output
                Feature = [Feature; curr_Feature];
            else
                if size(curr_Feature{1}{1},1) == 1 % Single Frequency Computation
                    curr_Feature_curated = NaN([length(curr_Feature{1}),length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
                    curr_feature_label_curated = cell([length(curr_Feature{1}),length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
                    
                else % Between Frequency Computation
                    feat_size = [];
                    for k = 1:length(size(curr_Feature{1}{1}{1}))
                        feat_size{k} = max(max(cell2mat(cellfun(@(x) size(x,k),curr_Feature{1}{1},'UniformOutput',0))));                        
                    end
                    feat_size = cell2mat(feat_size);
                    curr_Feature_curated = NaN([length(curr_Feature{1}),size(curr_Feature{1}{1}),feat_size]);
                    curr_Feature_label_curated = NaN([length(curr_Feature{1}),size(curr_Feature{1}{1}),feat_size]);
                end
                    
                for m = 1:length(curr_Feature{1}) % Number of Windows
                    curr_feature_label = '';
                    curr_feature_label = [curr_feature_label 'W' num2str(m)];
                    for n = 1:size(curr_Feature{1}{1},2) % Number of Frequency Windows
                        curr_feature_label = [curr_feature_label 'F' num2str(n)];
                        if size(curr_Feature{1}{m},1) == 1 % Single Frequency Computation
                            curr_Feature_curated(m,n,:,:) = curr_Feature{1}{m}{n};
                        else % Between Frequency Computation
                            for p = 1:size(curr_Feature{1}{1},1)
                                if ~isempty(curr_Feature{1}{m}{p,n}) curr_Feature_curated(m,p,n,:,:) = curr_Feature{1}{m}{p,n}; end    % TO FIX - need to add feat_size to this to make it compatible with features that are not a single value per pair of electrodes 
                            end
                        end
                    end
                end
                Feature{j} = curr_Feature_curated(:);
            end
        end 
        analyzedData = []; analyzedData.Feature = Feature; analyzedData.Feature_size = size(curr_Feature_curated); analyzedData.curr_Feature_curated = curr_Feature_curated;
        parsave_struct([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}], analyzedData, 0)
        % parsave([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}],Feature,Feature_size);
    end
    features_curated = true;

elseif ~isempty(Featurefiles_curated_dir)
    disp(['Features already curated ' newline]);
    features_curated = true;
    
elseif isempty(Featurefiles_dir)
    disp(['Features not computed yet ' newline]);
% else
%     disp('Features already curated \n');
%     features_curated = true;
end

if provide_final && features_curated
    final_Feature = []; % final_FeatureIDX = [];
    final_FeatureIDX = zeros(length(currFeatures_curated),2);
    for i = 1:length(currFeatures_curated)
        feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
        if isfield(feat_file,'analyzedData') curr_feat = cell2mat(feat_file.analyzedData)';
        else curr_feat = cell2mat(feat_file.Feature)'; end
        final_FeatureIDX(i,1) = length(final_Feature)+1; final_FeatureIDX(i,2) = length(final_Feature)+length(curr_feat);
        final_Feature = [final_Feature curr_feat];
        % final_FeatureIDX = [final_FeatureIDX ones(1,size(curr_feat,2))*i]; 
    end    
    varargout{1} = final_Feature; varargout{2} = final_FeatureIDX;
end