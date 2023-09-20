function Features_to_process = get_finalFeatureName(feature_names,featureVar_to_load)

% NEED to change so that the output is ordered according to the right order

FinalFeatureName = cell(size(feature_names));
Features_to_process = [];
switch featureVar_to_load
    case 'final'
        
        if sum(cell2mat(cellfun(@(x) strcmp(x,'COH'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'avg_coherence'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PAC'),feature_names,'UniformOutput',0))) compute_feat.compute_PAC = 1; Features_to_process{end+1} = 'arr_avgamp'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PLI'),feature_names,'UniformOutput',0))) compute_feat.compute_PLI = 1; Features_to_process{end+1} = 'PLI_z_score'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'dPLI'),feature_names,'UniformOutput',0))) compute_feat.compute_dPLI = 1; Features_to_process{end+1} = 'dPLI_z_score'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'STE'),feature_names,'UniformOutput',0))) compute_feat.compute_STE = 1; Features_to_process{end+1} = 'STE'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'Bspec'),feature_names,'UniformOutput',0))) compute_feat.compute_Bspec = 1; Features_to_process{end+1} = 'Bspec_features'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'CFC_SI'),feature_names,'UniformOutput',0))) compute_feat.compute_CFC_SI = 1; Features_to_process{end+1} = 'CFC_SI_mag'; end
        
    case 'full'
        if sum(cell2mat(cellfun(@(x) strcmp(x,'COH'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'avg_coherence'; Features_to_process{end+1} = 'all_coherence'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PAC'),feature_names,'UniformOutput',0))) compute_feat.compute_PAC = 1; Features_to_process{end+1} = 'arr_sortamp'; Features_to_process{end+1} = 'arr_plotamp'; Features_to_process{end+1} = 'arr_avgamp'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PLI'),feature_names,'UniformOutput',0))) compute_feat.compute_PLI = 1; Features_to_process{end+1} = 'PLI_z_score'; Features_to_process{end+1} = 'PLI'; Features_to_process{end+1} = 'PLIcorr'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'dPLI'),feature_names,'UniformOutput',0))) compute_feat.compute_dPLI = 1; Features_to_process{end+1} = 'dPLI_z_score'; Features_to_process{end+1} = 'dPLI'; Features_to_process{end+1} = 'dPLIcorr'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'STE'),feature_names,'UniformOutput',0))) compute_feat.compute_STE = 1; Features_to_process{end+1} = 'STE'; Features_to_process{end+1} = 'NSTE'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'Bspec'),feature_names,'UniformOutput',0))) compute_feat.compute_Bspec = 1; Features_to_process{end+1} = 'Bspec_features'; Features_to_process{end+1} = 'Bspec_features_raw_cell'; Features_to_process{end+1} = 'Bspec_features_real_cell'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'CFC_SI'),feature_names,'UniformOutput',0))) compute_feat.compute_CFC_SI = 1; Features_to_process{end+1} = 'CFC_SI_mag'; Features_to_process{end+1} = 'CFC_SI'; Features_to_process{end+1} = 'CFC_SI_theta'; end
end