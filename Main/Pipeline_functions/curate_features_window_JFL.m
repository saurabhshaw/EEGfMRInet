function [compute_feat, varargout] = curate_features_window_JFL(feature_names, featureVar_to_load, Featurefiles_basename, Featurefiles_directory, cell_output, provide_final)
% Features_to_process is a cell containing a string of all of the feature names to curate
% if mRMR_features = 0, then do not run 

% Determine if curate full feature set for each feature, or the final
% computed feature for each feature:
Features_to_process = [];
switch featureVar_to_load
    case 'final'
        if sum(cell2mat(cellfun(@(x) strcmp(x,'COH'),feature_names,'UniformOutput',0))) compute_feat.compute_COH = 1; Features_to_process{end+1} = 'avg_coherence'; end
        if sum(cell2mat(cellfun(@(x) strcmp(x,'PAC'),feature_names,'UniformOutput',0))) compute_feat.compute_PAC = 1; Features_to_process{end+1} = 'arr_plotamp'; end
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


%% Curate features from all available files:
Featurefiles_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_Epoch*.mat']);
if ~isempty(Featurefiles_dir)
    Featurefiles_names = {Featurefiles_dir(:).name}; curr_file_order = cell2mat(cellfun(@(x) str2num(x((strfind(x,'h') + 1):(strfind(x,'.') - 1))),Featurefiles_names,'UniformOutput',0));
    [~, curr_file_order_sorted_idx] = sort(curr_file_order);
    for i = 1:length(Features_to_process) % Removed parsave here
        Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i} '.mat']);
        if isempty(Featurefiles_curated_dir)
            Feature = []; Feature_labels = []; Feature_Epoch = [];
            for j = 1:length(curr_file_order_sorted_idx)
                
                curr_Epoch = curr_file_order_sorted_idx(j); curr_Epoch_name = curr_file_order(curr_Epoch);
                disp(['Running Feature ' Features_to_process{i} ' Epoch ' num2str(curr_Epoch)]);
                curr_file = load([Featurefiles_directory filesep Featurefiles_dir(curr_Epoch).name]);
                curr_file_cell = struct2cell(curr_file); curr_file_cell_var = fields(curr_file);
                curr_feature_idx = cell2mat(cellfun(@(x) strcmp(x,Features_to_process{i}), curr_file_cell_var,'UniformOutput',0));
                curr_Feature = curr_file_cell{curr_feature_idx};
                
                % Give feature output as cell or as a matrix:
                if cell_output
                    Feature = [Feature; curr_Feature];
                else
%                     if size(curr_Feature{1}{1},1) == 1 % Single Feature Computation
%                         curr_Feature_curated = NaN([length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
%                         curr_feature_label_curated = cell([length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
%                         
%                     else % Between Frequency Computation
%                         feat_size = [];
%                         for k = 1:length(size(curr_Feature{1}{1}{1}))
%                             feat_size{k} = max(max(cell2mat(cellfun(@(x) size(x,k),curr_Feature{1}{1},'UniformOutput',0))));
%                         end
%                         feat_size = cell2mat(feat_size);
%                         curr_Feature_curated = NaN([size(curr_Feature{1}{1}),feat_size]);
%                         curr_Feature_label_curated = NaN([size(curr_Feature{1}{1}),feat_size]);
%                     end
                    
                    curr_m_Feature = cell(1,length(curr_Feature{1})); curr_m_Feature_Epoch = cell(1,length(curr_Feature{1}));
                    parfor m = 1:length(curr_Feature{1}) % Number of Windows 
                        if size(curr_Feature{1}{1},1) == 1 % Single Feature Computation
                            curr_Feature_curated = NaN([length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
                            curr_feature_label_curated = cell([length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
                            
                        else % Between Frequency Computation
                            feat_size = [];
                            for k = 1:length(size(curr_Feature{1}{1}{1}))
                                feat_size{k} = max(max(cell2mat(cellfun(@(x) size(x,k),curr_Feature{1}{1},'UniformOutput',0))));
                            end
                            feat_size = cell2mat(feat_size);
                            curr_Feature_curated = NaN([size(curr_Feature{1}{1}),feat_size]);
                            curr_Feature_label_curated = NaN([size(curr_Feature{1}{1}),feat_size]);
                        end
                        curr_feature_label = '';
                        curr_feature_label = [curr_feature_label 'W' num2str(m)];
                        for n = 1:size(curr_Feature{1}{1},2) % Number of Frequency Windows
                            curr_feature_label = [curr_feature_label 'F' num2str(n)];
                            if size(curr_Feature{1}{m},1) == 1 % Single Frequency Computation
                                curr_Feature_curated(n,:,:) = curr_Feature{1}{m}{n};
                            else % Between Frequency Computation
                                for p = 1:size(curr_Feature{1}{1},1)
                                    if ~isempty(curr_Feature{1}{m}{p,n}) curr_Feature_curated(p,n,:,:) = curr_Feature{1}{m}{p,n}; end
                                end
                            end
                        end
                        %curr_m_Feature = [curr_m_Feature; curr_Feature_curated(:)'];
                        %curr_m_Feature_Epoch = [curr_m_Feature_Epoch; curr_Epoch_name];
                        curr_m_Feature{m} = curr_Feature_curated(:);
                        curr_m_Feature_Epoch{m} = curr_Epoch_name;
                    end
                    % Feature = curr_m_Feature'; Feature_Epoch = curr_m_Feature_Epoch;
                    Feature{j} = cell2mat(curr_m_Feature); Feature_Epoch{j} = cell2mat(curr_m_Feature_Epoch);
                end
            end
            
            % save([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}],'Feature','Feature_Epoch')
            parsave([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}],Feature,Feature_Epoch);
        else
            disp([Features_to_process{i} ' already curated']);
        end        
    end
else
    disp('Features already curated');
end

if provide_final 
    final_Feature = [];
    for i = 1:length(Features_to_process)
        feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i} '.mat']);
        final_Feature = [final_Feature cell2mat(feat_file.analyzedData)'];
        % varargout{i} = final_Feature;
    end    
    varargout{1} = final_Feature;
end









%% Old code based on AllEpochs curation:
% fullFeatures_dir = dir([curr_dir filesep 'EEG_Features' filesep 'Rev_*AllEpochs.mat']);
%             if ~isempty(fullFeatures_dir)
%                 [~,systemview] = memory;
%                 if systemview.PhysicalMemory.Available >= fullFeatures_dir(1).bytes
%                     load([curr_dir filesep 'EEG_Features' filesep fullFeatures_dir(1).name]);
%                     Features_matfile_used = 0;
%                 else
%                     Features_matFile = matfile([curr_dir filesep 'EEG_Features' filesep fullFeatures_dir(1).name]);
%                     Features_matfile_used = 1;
%                 end
%             end




% switch featureVar_to_load
%     case 'final'
%         if ~Features_matfile_used
%             if compute_COH final_Features_cell = cellfun(@(x,y) cat(y, x.avg_coherence,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             
%             
%         else
%             if compute_COH final_Features_cell = cellfun(@(x,y) cat(y, x.avg_coherence,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             if compute_PAC final_Features_cell = cellfun(@(x,y) cat(y, x.arr_plotamp,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             if compute_PLI final_Features_cell = cellfun(@(x,y) cat(y, x.PLI_z_score,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             if compute_dPLI final_Features_cell = cellfun(@(x,y) cat(y, x.dPLI_z_score,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             if compute_STE final_Features_cell = cellfun(@(x,y) cat(y, x.STE,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             if compute_Bspec final_Features_cell = cellfun(@(x,y) cat(y, x.Bspec_features,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%             if compute_CFC_SI final_Features_cell = cellfun(@(x,y) cat(y, x.CFC_SI_mag,2) ,Features_matFile.Features,final_Features_cell,'UniformOutput',0); end
%         end
%         
%     case 'full'
%         if compute_COH Features_matFile.Features{:}.all_coherence = all_coherence; Features_matFile.avg_coherence = avg_coherence; end
%         if compute_PAC Features_matFile.arr_sortamp = arr_sortamp; Features_matFile.arr_plotamp = arr_plotamp; Features_matFile.arr_avgamp = arr_avgamp; end
%         if compute_PLI Features_matFile.PLI = PLI; Features_matFile.PLIcorr = PLIcorr; Features_matFile.PLI_z_score = PLI_z_score; end
%         if compute_dPLI Features_matFile.dPLI = dPLI; Features_matFile.dPLIcorr = dPLIcorr; Features_matFile.dPLI_z_score = dPLI_z_score; end
%         if compute_STE Features_matFile.STE = STE; Features_matFile.NSTE = NSTE; end
%         if compute_Bspec Features_matFile.Bspec_features_raw_cell = Bspec_features_raw_cell; Features_matFile.Bspec_features_real_cell = Bspec_features_real_cell; Features_matFile.Bspec_features = Bspec_features; end
%         if compute_CFC_SI Features_matFile.CFC_SI = CFC_SI; Features_matFile.CFC_SI_mag = CFC_SI_mag; Features_matFile.CFC_SI_theta = CFC_SI_theta; end
%         
% end