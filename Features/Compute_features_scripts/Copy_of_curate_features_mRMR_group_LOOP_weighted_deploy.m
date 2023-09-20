% This function runs one curate_features_mRMR_group run
function curate_features_mRMR_group_LOOP_weighted_deploy(i,Featurefiles_directory_ALL,curr_Featurefiles_basename_ALL,YY_final_ALL,weights,name_suffix,output_base_path_data,study_name,max_features,CONN_cfg,varargin)

%% Find curated features:
% curr_Featurefiles_basename = strsplit(Featurefiles_basename,'_CLASS');
% curr_Featurefiles_basename = curr_Featurefiles_basename{1};
Featurefiles_curated_dir = dir([Featurefiles_directory_ALL{1} filesep curr_Featurefiles_basename_ALL{1} '_AllEpochs_*.mat']);
% Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

%% Iterative mRMR:                
% curr_labels_mRMR = YY_final; % Get current labels for mRMR

wind_dataset_mRMR = []; wind_feature_labels_mRMR = [];
wind_output_features = cell(1,length(currFeatures_curated)); wind_output_scores = cell(1,length(currFeatures_curated));

% for i = 1:length(currFeatures_curated) % Number of features
%% i is defined at input

% Create matfile cell:
% feat_file_matfiles = cellfun(@(x,y)matfile([x filesep y '_AllEpochs_' currFeatures_curated{i} '.mat']),Featurefiles_directory_ALL,curr_Featurefiles_basename_ALL,'un',0);
groupFeature_matfile = matfile([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'AllData_' currFeatures_curated{i}]);
num_feat_timepnts = size(groupFeature_matfile,'Feature');

% feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
% feat_file = load([Featurefiles_directory filesep curr_Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);

% Convert linear indices to nested index:
% curr_Feature_curated = cellfun(@(x)reshape(x,[feat_file.Feature_size]),feat_file.Feature,'un',0);
% curr_Feature_curated = cellfun(@(x)cellfun(@(y)reshape(y,[feat_file_Feature_size]),x.Feature,'un',0),feat_file_matfiles,'un',0);

% Get feature sizes:
% innermost_feature_size = feat_file.Feature_size(end-1:end);
% feat_file_Feature_size = feat_file.Feature_size;
% feat_file_Feature_size = feat_file_matfiles{1}.Feature_size;
feat_file_Feature_size = groupFeature_matfile.Feature_size;
innermost_feature_size = feat_file_Feature_size(end-1:end);

if isempty(varargin) % This is if the all the timepoints are to be used 
    temp_cell = groupFeature_matfile.Feature(1,1,1:num_feat_timepnts(end));
    % temp_cell = cellfun(temp_cell,'un',0);
    curr_dataset_mRMR_IDX = cellfun(@(x)~isempty(squeeze(x)),temp_cell);

else % This is if a very specific set of timepoints are to be used (specific participant/session)
    curr_dataset_mRMR_IDX = varargin{1};
end

% Define the labels:
if iscell(YY_final_ALL) 
    curr_labels_mRMR = cat(2,YY_final_ALL{:}); % Get current labels for mRMR
    curr_labels_mRMR_final = curr_labels_mRMR(curr_dataset_mRMR_IDX);
else
    curr_labels_mRMR = YY_final_ALL;
    curr_labels_mRMR_final = curr_labels_mRMR;
end

% Splitting it up here since parfor needs to be at separate levels for
% single frequency vs double frequency features:
if length(feat_file_Feature_size) == 4 % Single Frequency Computation
    freq_dataset_mRMR = cell(1,feat_file_Feature_size(1)); freq_feature_labels_mRMR = cell(1,feat_file_Feature_size(1));
    % freq_dataset_mRMR = []; freq_feature_labels_mRMR = [];
    freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
    
    parfor m = 1:feat_file_Feature_size(1) % Number of Windows = WAS PARFOR
        % for m = 1:4 % Number of Windows
        elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
        % elec_dataset_mRMR = cell(1,feat_file.Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file.Feature_size(2));
        elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
        
        tic
        for n = 1:feat_file_Feature_size(2) % Number of Frequency Windows
            
            % Get current dataset for mRMR:
            % Get the linear indices of the appropriate features:
            %                 [X,Y] = meshgrid(1:innermost_feature_size(1),1:innermost_feature_size(2)); X = X(:); Y = Y(:);
            %                 currIDX = sub2ind(feat_file_Feature_size,repmat(m,[length(X), 1]),repmat(n,[length(X), 1]),X,Y);
            %                 curr_dataset_mRMR = []; curr_dataset_mRMR = cell(size(feat_file_matfiles));
            %                 parfor tt = 1:length(feat_file_matfiles)
            %                     tic;
            %                     temp_Feature = cell2mat(cellfun(@(x)x(currIDX),feat_file_matfiles{tt}.Feature,'un',0))';
            %                     curr_dataset_mRMR{tt} = temp_Feature;
            %                     % curr_dataset_mRMR = cat(1,curr_dataset_mRMR,temp_Feature);
            %                     toc
            %                     disp(['tt = ' num2str(tt) '/' num2str(length(feat_file_matfiles))]);
            % %                     tic
            % %                     num_timepts = size(feat_file_matfiles{tt},'Feature'); num_timepts = num_timepts(2);
            % %                     for mm = 1:num_timepts
            % %                         temp_Feature = feat_file_matfiles{tt}.Feature(1,mm);
            % %                         curr_dataset_mRMR = cat(1,curr_dataset_mRMR,temp_Feature{1}(currIDX)');
            % %                         disp(['tt = ' num2str(tt) '/' num2str(length(feat_file_matfiles)) ' & mm = ' num2str(mm) '/' num2str(num_timepts)])
            % %                     end
            % %                     toc
            %                 end
            %                 curr_dataset_mRMR = cat(1,curr_dataset_mRMR{:});
            %                 % curr_feature_size = feat_file.Feature_size(end-1:end);
            %                 % curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,n,:,:)),curr_Feature_curated,'un',0);
            %                 % curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
            
            
            curr_dataset_mRMR = groupFeature_matfile.Feature(m,n,1:num_feat_timepnts(end)); % toc
            curr_dataset_mRMR = squeeze(curr_dataset_mRMR); expected_length = innermost_feature_size(1)*innermost_feature_size(2);
            % curr_dataset_mRMR_IDX = cellfun(@(x)length(x(:)'),curr_dataset_mRMR) == expected_length;
            curr_dataset_mRMR = curr_dataset_mRMR(curr_dataset_mRMR_IDX);
            curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:)',curr_dataset_mRMR,'un',0));
            
            % Run mRMR at this level:
            [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_weighted_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR_final,weights,innermost_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR);
            % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_weighted_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,weights,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n});
            
            disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBand ' num2str(n)]);
            
        end
        toc
        
        % Run mRMR at this level:
        % elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
        % elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
        curr_feature_size = [feat_file_Feature_size(2) size(elec_output_features{1},2)];
        % [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR,freq_feature_labels_mRMR] = mRMR_weighted_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,freq_feature_labels_mRMR,elec_feature_labels_mRMR);
        [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR{m},freq_feature_labels_mRMR{m}] = mRMR_weighted_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR_final,weights,curr_feature_size,max_features,freq_dataset_mRMR{m},freq_feature_labels_mRMR{m},elec_feature_labels_mRMR);
        
    end
    
    save([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'mRMRiterate_' currFeatures_curated{i} '_' CONN_cfg.class_types name_suffix],'-v7.3');
    
    freq_dataset_mRMR = cat(2,freq_dataset_mRMR{:});
    freq_feature_labels_mRMR = cat(2,freq_feature_labels_mRMR{:});
    
    
else % Between frequency computation
    % freq_dataset_mRMR = cell(1,feat_file_Feature_size(1)); freq_feature_labels_mRMR = cell(1,feat_file.Feature_size(1));
    freq_dataset_mRMR = []; freq_feature_labels_mRMR = [];
    freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
    
    for m = 1:feat_file_Feature_size(1) % Number of Windows
        % for m = 1:4 % Number of Windows
        
        % elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
        elec_dataset_mRMR = cell(1,feat_file_Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file_Feature_size(2));
        elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
        
        % curr_dataset_mRMR_parfor = cellfun(@(x)squeeze(x(m,:,:,:,:)),curr_Feature_curated,'un',0);
        
        tic
        % Run one less outer loop for frequency due to the n < p
        % computation used for cross-frequency features - this causes
        % all n-p combinations of the last n to be empty
        % curr_dataset_mRMR_IDX = nan((feat_file_Feature_size(2)-1),feat_file_Feature_size(3),length(curr_dataset_mRMR_parfor));
        parfor n = 1:(feat_file_Feature_size(2)-1) % Number of Frequency Windows
            
            innerFreq_dataset_mRMR = []; innerFreq_feature_labels_mRMR = []; innerFreq_loopIDX_mRMR = [];
            % innerFreq_dataset_mRMR = cell(1,feat_file.Feature_size(3)); innerFreq_feature_labels_mRMR = cell(1,feat_file.Feature_size(3));
            % elec_dataset_mRMR = cell(1,feat_file.Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file.Feature_size(2));
            innerFreq_output_features = cell(1,feat_file_Feature_size(3)); innerFreq_output_scores = cell(1,feat_file_Feature_size(3));
            
            % curr_dataset_mRMR_parfor = cellfun(@(x)squeeze(x(m,:,n,:,:)),curr_Feature_curated,'un',0);
            curr_dataset_mRMR_parfor = groupFeature_matfile.Feature(m,n,1:num_feat_timepnts(end)); % toc
            curr_dataset_mRMR_parfor = squeeze(curr_dataset_mRMR_parfor); expected_length = innermost_feature_size(1)*innermost_feature_size(2);
            
            
            for p = 1:feat_file_Feature_size(3)
                % flip p and n index positions because this was not
                % kept consistent from the single frequency computation
                % where n is the second index - fix this by indexing
                % curr_Feature_curated(m,n,p,:,:) in
                % curate_features_deploy in future
                % But since the number of features on both levels are
                % the same - n,p works
                
                % if ~isempty(curr_Feature{1}{m}{p,n}) curr_Feature_curated(m,p,n,:,:) = curr_Feature{1}{m}{p,n}; end    % TO FIX - need to add feat_size to this to make it compatible with features that are not a single value per pair of electrodes
                if n < p % This is how features were computed for cross-frequency features
                    
                    % Get indices:
                    % Get the linear indices of the appropriate features:
                    %                         [X,Y] = meshgrid(1:innermost_feature_size(1),1:innermost_feature_size(2)); X = X(:); Y = Y(:);
                    %                         currIDX = sub2ind(feat_file_Feature_size,repmat(m,[length(X), 1]),repmat(n,[length(X), 1]),repmat(p,[length(X), 1]),X,Y);
                    %
                    %                         % Get current dataset for mRMR:
                    %                         % curr_feature_size = feat_file.Feature_size(end-1:end);
                    %                         % curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,p,n,:,:)),curr_Feature_curated,'un',0);
                    %                         curr_dataset_mRMR = cellfun(@(x)squeeze(x(n,p,:,:)),curr_dataset_mRMR_parfor,'un',0);
                    %                         curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
                    
                    % curr_dataset_mRMR_IDX = cellfun(@(x)~isempty(x),curr_dataset_mRMR_parfor);
                    curr_dataset_mRMR = cellfun(@(x)squeeze(x(p,:,:)),curr_dataset_mRMR_parfor(curr_dataset_mRMR_IDX),'un',0);
                    % curr_dataset_mRMR_IDX = cellfun(@(x)length(x(:)'),curr_dataset_mRMR_parfor) == expected_length;
                    curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:)',curr_dataset_mRMR,'un',0));
                    
                    % Run mRMR at this level:
                    [innerFreq_output_features{p}, innerFreq_output_scores{p},innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR] = mRMR_weighted_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR_final,weights,innermost_feature_size,max_features,innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR);
                    % [innerFreq_output_features{p}, innerFreq_output_scores{p},innerFreq_dataset_mRMR{p},innerFreq_feature_labels_mRMR{p}] = mRMR_weighted_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,weights,curr_feature_size,max_features,innerFreq_dataset_mRMR{p},innerFreq_feature_labels_mRMR{p});
                    
                    innerFreq_loopIDX_mRMR = [innerFreq_loopIDX_mRMR repmat([p],1,size(innerFreq_output_features{p},2))];
                    
                    % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_weighted_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,weights,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n});
                end
                disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBands ' num2str(n) '-' num2str(p)]);
                
            end
            
            % Run mRMR at this level:
            % innerFreq_dataset_mRMR = cat(2,innerFreq_dataset_mRMR{:});
            % innerFreq_feature_labels_mRMR = cat(2,innerFreq_feature_labels_mRMR{:});
            curr_feature_size = [feat_file_Feature_size(3) size(innerFreq_output_features{end},2)];
            [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_weighted_iterate_loop(innerFreq_dataset_mRMR,curr_labels_mRMR_final,weights,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n},innerFreq_feature_labels_mRMR,innerFreq_loopIDX_mRMR);
            % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_weighted_iterate_loop(innerFreq_dataset_mRMR,curr_labels_mRMR,weights,curr_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR,innerFreq_feature_labels_mRMR);
            
        end
        toc
        
        % save([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'mRMRiterate_' currFeatures_curated{i} '_' CONN_cfg.class_types num2str(m)],'-v7.3','elec_output_features', 'elec_output_scores','elec_dataset_mRMR','elec_feature_labels_mRMR');
        save([output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'mRMRiterate_' currFeatures_curated{i} '_' CONN_cfg.class_types '_CHECKPOINT_' num2str(m) name_suffix],'-v7.3');
        
        % Run mRMR at this level:
        elec_loopIDX_mRMR = [];
        for temp_idx = 1:length(elec_output_features) elec_loopIDX_mRMR = [elec_loopIDX_mRMR repmat([temp_idx],[1 size(elec_output_features{temp_idx},2)])]; end
        
        elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
        elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
        curr_feature_size = [feat_file_Feature_size(2) size(elec_output_features{1},2)];
        [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR,freq_feature_labels_mRMR] = mRMR_weighted_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR_final,weights,curr_feature_size,max_features,freq_dataset_mRMR,freq_feature_labels_mRMR,elec_feature_labels_mRMR,elec_loopIDX_mRMR);
        % [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR{m},freq_feature_labels_mRMR{m}] = mRMR_weighted_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,weights,curr_feature_size,max_features,freq_dataset_mRMR{m},freq_feature_labels_mRMR{m},elec_feature_labels_mRMR);
        
    end
end

% Run mRMR at this level:

curr_feature_size = [feat_file_Feature_size(1) size(freq_output_features{1},2)];
% [wind_output_features{i}, wind_output_scores{i},wind_dataset_mRMR,wind_feature_labels_mRMR] = mRMR_weighted_iterate_loop(freq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,weights,max_features,freq_dataset_mRMR,wind_feature_labels_mRMR,freq_feature_labels_mRMR);
[wind_output_features{i}, wind_output_scores{i},wind_dataset_mRMR,wind_feature_labels_mRMR] = mRMR_weighted_iterate_loop(freq_dataset_mRMR,curr_labels_mRMR_final,weights,curr_feature_size,max_features,wind_dataset_mRMR,wind_feature_labels_mRMR,freq_feature_labels_mRMR);

% end

final_dataset_mRMR = wind_dataset_mRMR;
final_feature_labels_mRMR = wind_feature_labels_mRMR;

nestedFeature_struct = [];
nestedFeature_struct.wind_output_features = wind_output_features;
nestedFeature_struct.wind_output_scores = wind_output_scores;
nestedFeature_struct.freq_output_features = freq_output_features;
nestedFeature_struct.freq_output_scores = freq_output_scores;
nestedFeature_struct.freq_dataset_mRMR = freq_dataset_mRMR;
nestedFeature_struct.freq_feature_labels_mRMR = freq_feature_labels_mRMR;
% nestedFeature_struct.elec_output_features = elec_output_features;
% nestedFeature_struct.elec_dataset_mRMR = elec_dataset_mRMR;
% nestedFeature_struct.elec_feature_labels_mRMR = elec_feature_labels_mRMR;
% if exist('innerFreq_output_features') nestedFeature_struct.innerFreq_output_features = innerFreq_output_features; end
% if exist('innerFreq_output_scores') nestedFeature_struct.innerFreq_output_scores = innerFreq_output_scores; end
% if exist('innerFreq_dataset_mRMR') nestedFeature_struct.innerFreq_dataset_mRMR = innerFreq_dataset_mRMR; end
% if exist('innerFreq_feature_labels_mRMR') nestedFeature_struct.innerFreq_feature_labels_mRMR = innerFreq_feature_labels_mRMR; end

% Save the output data:
% 'Features','Feature_labels_mRMR','Feature_mRMR_order'
Results_outputDir = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'LeaveOneOut_IND_Features'];
if isempty(dir(Results_outputDir)) mkdir(Results_outputDir); end
save([Results_outputDir filesep study_name '_' currFeatures_curated{i} '_mRMRiterateGroupResults_' CONN_cfg.class_types name_suffix],'final_dataset_mRMR','final_feature_labels_mRMR','curr_labels_mRMR','curr_dataset_mRMR_IDX','currFeatures_curated','nestedFeature_struct','-v7.3');
