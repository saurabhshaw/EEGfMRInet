function curate_features_mRMR_checkpoint_global_deploy(input_mat)
% [final_dataset_mRMR,final_feature_labels_mRMR,currFeatures_curated,varargout] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features)

% Read in data:
% load(input_mat);

max_features = max_features;


%% Create Group Matfile Array:
% sub_dir; sub_dir_mod; output_base_path_data; EEGfMRI_corrIDX; CONN_data

for ii = 2:length(sub_dir) % GRAHAM_PARFOR-1
    skip_analysis = 0;
    dataset_to_use = [sub_dir(ii).name];
    dataset_name = [sub_dir_mod(ii).PID];
    curr_dir = [output_base_path_data filesep dataset_to_use];
    
    temp_idx = ~isnan(EEGfMRI_corrIDX(ii,:)); % Check if this task block has been processed by CONN:

    for m = find(temp_idx) % GRAHAM_PARFOR-2
        
        fprintf(['\n ***************************** Processing Subject ' sub_dir_mod(ii).PID ' Session ' sub_dir_mod(ii).SID ' Run ' runs_to_include{jj} ' ***************************** \n']);
        
        % missing_curate_features_mRMR = 0;
        
        % Obtain the label vector:
        % Its already in CONN_data
        curr_CONN_IDX = EEGfMRI_corrIDX(ii,m);
        YY_final = cell2mat(CONN_data.fMRI_labels_selected_window_avg_thresh{ii-1}{curr_CONN_IDX});  % NOTE:only because this is excluding the first subject
        YY_final_continuous = (CONN_data.fMRI_labels_selected_window_avg{ii-1}{curr_CONN_IDX}); YY_final_continuous = cat(1,YY_final_continuous{:}); % NOTE:only because this is excluding the first subject
        YY_final_continuous_thresh = double(YY_final_continuous >= CONN_cfg.threshold);
        
        % Obtain features:
        % nclassesIdx = randperm(length(YY_final));
        % [Features,Feature_labels_mRMR,Feature_mRMR_order] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features);
        % save([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults'],'Features','Feature_labels_mRMR','Feature_mRMR_order');
        task_dir = [curr_dir filesep 'Task_block_' num2str(m)];
        Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(m)];
        Featurefiles_basename = ['Rev_' curr_dataset_name];
        
        load([Featurefiles_directory filesep curr_Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
        
    end
end

%curr_filedir = dir([curr_dir filesep '*.set']);
%curr_file = [curr_dir filesep dataset_name '.set'];


% Find curated features:
curr_Featurefiles_basename = strsplit(Featurefiles_basename,'_CLASS');
curr_Featurefiles_basename = curr_Featurefiles_basename{1};
Featurefiles_curated_dir = dir([Featurefiles_directory filesep curr_Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

% Find the last checkpoint:
last_checkpoint = 0;
dir_checkpoints = dir([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults' '_CHECKPOINT' '*.mat']);
if isempty(dir_checkpoints) % If there are no previous checkpoints
    if isempty(dir([Featurefiles_directory filesep 'checkpoints']))
        mkdir([Featurefiles_directory filesep 'checkpoints']); 
    end
    last_checkpoint_data = [];
    last_checkpoint_data.i = 1;
    last_checkpoint_data.m = 1;
else
    % dir_checkpoints = dir([Featurefiles_directory filesep 'checkpoints' filesep '*.mat']);
    dir_checkpoints_num = cellfun(@(x)strsplit(x,{'_CHECKPOINT','.mat'}),{dir_checkpoints.name},'un',0);
    dir_checkpoints_num = cellfun(@(x)str2num(x{2}),dir_checkpoints_num);
    [dir_checkpoints_num_sorted,dir_checkpoints_num_sortedIDX] = sort(dir_checkpoints_num,'ascend');
    last_checkpoint = dir_checkpoints_num_sorted(end);
    try % In case it is unreadable
        last_checkpoint_data = load([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults' '_CHECKPOINT' num2str(last_checkpoint)]);
    catch
        last_checkpoint = last_checkpoint - 1;
        last_checkpoint_data = load([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults' '_CHECKPOINT' num2str(last_checkpoint)]);
    end
        
    % Delete older checkpoints, keeping only the last 2:
    if length(dir_checkpoints) > 2
        idx_toDelete = dir_checkpoints_num_sortedIDX(1:end-2);
        cellfun(@(x)delete([Featurefiles_directory filesep 'checkpoints' filesep x]),{dir_checkpoints(idx_toDelete).name});
    end
        
end

%% Iterative mRMR:                
curr_labels_mRMR = YY_final; % Get current labels for mRMR

% Initialize mRMR variables for this level:
if last_checkpoint == 0
    wind_dataset_mRMR = []; wind_feature_labels_mRMR = [];
    wind_output_features = cell(1,length(currFeatures_curated)); wind_output_scores = cell(1,length(currFeatures_curated));
else    
    wind_dataset_mRMR = last_checkpoint_data.wind_dataset_mRMR;
    wind_feature_labels_mRMR = last_checkpoint_data.wind_feature_labels_mRMR;
    wind_output_features = last_checkpoint_data.wind_output_features;
    wind_output_scores = last_checkpoint_data.wind_output_scores;
end    
    
for i = last_checkpoint_data.i:length(currFeatures_curated) % Number of features
    feat_file = load([Featurefiles_directory filesep curr_Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
    
    % Convert linear indices to nested index:
    curr_Feature_curated = cellfun(@(x)reshape(x,[feat_file.Feature_size]),feat_file.Feature,'un',0);
    
    % Get feature sizes:
    innermost_feature_size = feat_file.Feature_size(end-1:end);
    feat_file_Feature_size = feat_file.Feature_size;
    
    % Splitting it up here since parfor needs to be at separate levels for
    % single frequency vs double frequency features:
    if length(feat_file_Feature_size) == 4 % Single Frequency Computation
        
        % Initialize mRMR variables for this level:
        if last_checkpoint == 0
            % freq_dataset_mRMR = cell(1,feat_file_Feature_size(1)); freq_feature_labels_mRMR = cell(1,feat_file_Feature_size(1));
            freq_dataset_mRMR = []; freq_feature_labels_mRMR = [];
            freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
            windows_to_process = 1:feat_file_Feature_size(1);
        else
            freq_dataset_mRMR = last_checkpoint_data.freq_dataset_mRMR;
            freq_feature_labels_mRMR = last_checkpoint_data.freq_feature_labels_mRMR;
            freq_output_features = last_checkpoint_data.freq_output_features;
            freq_output_scores = last_checkpoint_data.freq_output_scores;
            
            freq_output_features_finished = cellfun(@isempty,freq_output_features); freq_output_features_finished = find(~freq_output_features_finished);
            windows_to_process = setdiff(1:feat_file_Feature_size(1),freq_output_features_finished);
        end      
        
        for m = windows_to_process % Number of Windows (was parfor)
            % for m = 1:4 % Number of Windows
            % elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
            elec_dataset_mRMR = cell(1,feat_file.Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file.Feature_size(2));
            elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
            
            tic
            parfor n = 1:feat_file_Feature_size(2) % Number of Frequency Windows
                
                % Get current dataset for mRMR:
                % curr_feature_size = feat_file.Feature_size(end-1:end);
                curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,n,:,:)),curr_Feature_curated,'un',0);
                curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
                
                % Run mRMR at this level:
                % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,innermost_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR);
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,innermost_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n});
                
                disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBand ' num2str(n)]);
                
            end
            toc
            
            % Run mRMR at this level:
            elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
            elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
            curr_feature_size = [feat_file_Feature_size(2) size(elec_output_features{1},2)];
            [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR,freq_feature_labels_mRMR] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,freq_feature_labels_mRMR,elec_feature_labels_mRMR);
            % [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR{m},freq_feature_labels_mRMR{m}] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR{m},freq_feature_labels_mRMR{m},elec_feature_labels_mRMR);
            
            % Save checkpoint:
            last_checkpoint = last_checkpoint + 1;
            save([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults' '_CHECKPOINT' num2str(last_checkpoint)],'i','m',...
                'wind_output_features', 'wind_output_scores','wind_dataset_mRMR','wind_feature_labels_mRMR',...
                'freq_output_features', 'freq_output_scores','freq_dataset_mRMR','freq_feature_labels_mRMR',...
                'elec_output_features', 'elec_output_scores','elec_dataset_mRMR','elec_feature_labels_mRMR');
            
            
        end
        
        % freq_dataset_mRMR = cat(2,freq_dataset_mRMR{:});
        % freq_feature_labels_mRMR = cat(2,freq_feature_labels_mRMR{:});
        
%         % Save checkpoint:
%         last_checkpoint = last_checkpoint + 1;
%         save([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults_individual' ind_name '_CHECKPOINT' num2str(last_checkpoint)],'i','m',...
%             'wind_output_features', 'wind_output_scores','wind_dataset_mRMR','wind_feature_labels_mRMR',...
%             'freq_output_features', 'freq_output_scores','freq_dataset_mRMR','freq_feature_labels_mRMR',...
%             'elec_output_features', 'elec_output_scores','elec_dataset_mRMR','elec_feature_labels_mRMR');
%         
        
    else % Between frequency computation
        if last_checkpoint == 0
            % freq_dataset_mRMR = cell(1,feat_file_Feature_size(1)); freq_feature_labels_mRMR = cell(1,feat_file.Feature_size(1));
            freq_dataset_mRMR = []; freq_feature_labels_mRMR = [];
            freq_output_features = cell(1,feat_file_Feature_size(1)); freq_output_scores = cell(1,feat_file_Feature_size(1));
            windows_to_process = 1:feat_file_Feature_size(1);
        else
            freq_dataset_mRMR = last_checkpoint_data.freq_dataset_mRMR;
            freq_feature_labels_mRMR = last_checkpoint_data.freq_feature_labels_mRMR;
            freq_output_features = last_checkpoint_data.freq_output_features;
            freq_output_scores = last_checkpoint_data.freq_output_scores;
            
            freq_output_features_finished = cellfun(@isempty,freq_output_features); freq_output_features_finished = find(~freq_output_features_finished);
            windows_to_process = setdiff(1:feat_file_Feature_size(1),freq_output_features_finished);
        end
        
        for m = windows_to_process % Number of Windows
        % for m = last_checkpoint_data.m:feat_file_Feature_size(1) % Number of Windows
            % for m = 1:4 % Number of Windows
            
            % elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
            elec_dataset_mRMR = cell(1,feat_file_Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file_Feature_size(2));
            elec_output_features = cell(1,feat_file_Feature_size(2)); elec_output_scores = cell(1,feat_file_Feature_size(2));
            
            curr_dataset_mRMR_parfor = cellfun(@(x)squeeze(x(m,:,:,:,:)),curr_Feature_curated,'un',0);

            tic
            % Run one less outer loop for frequency due to the n < p
            % computation used for cross-frequency features - this causes
            % all n-p combinations of the last n to be empty
            parfor n = 1:(feat_file_Feature_size(2)-1) % Number of Frequency Windows
                
                innerFreq_dataset_mRMR = []; innerFreq_feature_labels_mRMR = []; innerFreq_loopIDX_mRMR = [];
                % innerFreq_dataset_mRMR = cell(1,feat_file.Feature_size(3)); innerFreq_feature_labels_mRMR = cell(1,feat_file.Feature_size(3));
                % elec_dataset_mRMR = cell(1,feat_file.Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file.Feature_size(2));
                innerFreq_output_features = cell(1,feat_file_Feature_size(3)); innerFreq_output_scores = cell(1,feat_file_Feature_size(3));

                % curr_dataset_mRMR_parfor = cellfun(@(x)squeeze(x(m,:,n,:,:)),curr_Feature_curated,'un',0);

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
                        % Get current dataset for mRMR:
                        % curr_feature_size = feat_file.Feature_size(end-1:end);
                        % curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,p,n,:,:)),curr_Feature_curated,'un',0);
                        curr_dataset_mRMR = cellfun(@(x)squeeze(x(n,p,:,:)),curr_dataset_mRMR_parfor,'un',0);
                        curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
                        
                        % Run mRMR at this level:
                        [innerFreq_output_features{p}, innerFreq_output_scores{p},innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,innermost_feature_size,max_features,innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR);
                        % [innerFreq_output_features{p}, innerFreq_output_scores{p},innerFreq_dataset_mRMR{p},innerFreq_feature_labels_mRMR{p}] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,innerFreq_dataset_mRMR{p},innerFreq_feature_labels_mRMR{p});
                        
                        innerFreq_loopIDX_mRMR = [innerFreq_loopIDX_mRMR repmat([p],1,size(innerFreq_output_features{p},2))];
                        
                        % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n});
                    end
                    disp(['Finished mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBands ' num2str(n) '-' num2str(p)]);
                    
                end
                
                % Run mRMR at this level:
                % innerFreq_dataset_mRMR = cat(2,innerFreq_dataset_mRMR{:});
                % innerFreq_feature_labels_mRMR = cat(2,innerFreq_feature_labels_mRMR{:});
                curr_feature_size = [feat_file_Feature_size(3) size(innerFreq_output_features{end},2)];
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_iterate_loop(innerFreq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n},innerFreq_feature_labels_mRMR,innerFreq_loopIDX_mRMR);
                % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_iterate_loop(innerFreq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR,innerFreq_feature_labels_mRMR);
                
            end
            toc
            
            % Run mRMR at this level:
            elec_loopIDX_mRMR = [];
            for temp_idx = 1:length(elec_output_features) elec_loopIDX_mRMR = [elec_loopIDX_mRMR repmat([temp_idx],[1 size(elec_output_features{temp_idx},2)])]; end
            
            elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
            elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
            curr_feature_size = [feat_file_Feature_size(2) size(elec_output_features{1},2)];
            [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR,freq_feature_labels_mRMR] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,freq_feature_labels_mRMR,elec_feature_labels_mRMR,elec_loopIDX_mRMR);
            % [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR{m},freq_feature_labels_mRMR{m}] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR{m},freq_feature_labels_mRMR{m},elec_feature_labels_mRMR);
            
            % Save checkpoint:
            last_checkpoint = last_checkpoint + 1;
            save([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults' '_CHECKPOINT' num2str(last_checkpoint)],'i','m',...
                'wind_output_features', 'wind_output_scores','wind_dataset_mRMR','wind_feature_labels_mRMR',...
                'freq_output_features', 'freq_output_scores','freq_dataset_mRMR','freq_feature_labels_mRMR',...
                'elec_output_features', 'elec_output_scores','elec_dataset_mRMR','elec_feature_labels_mRMR');

        end
    end
    
    % Run mRMR at this level:

    curr_feature_size = [feat_file_Feature_size(1) size(freq_output_features{1},2)];
    % [wind_output_features{i}, wind_output_scores{i},wind_dataset_mRMR,wind_feature_labels_mRMR] = mRMR_iterate_loop(freq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,wind_feature_labels_mRMR,freq_feature_labels_mRMR);
    [wind_output_features{i}, wind_output_scores{i},wind_dataset_mRMR,wind_feature_labels_mRMR] = mRMR_iterate_loop(freq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,wind_dataset_mRMR,wind_feature_labels_mRMR,freq_feature_labels_mRMR);

%     % Save checkpoint:
%     last_checkpoint = last_checkpoint + 1;
%     save([Featurefiles_directory filesep 'checkpoints' filesep Featurefiles_basename '_mRMRiterateResults_individual' ind_name '_CHECKPOINT' num2str(last_checkpoint)],...
%         'i','m','n','p','wind_output_features', 'wind_output_scores','wind_dataset_mRMR','wind_feature_labels_mRMR');

end

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
save([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults'],'final_dataset_mRMR','final_feature_labels_mRMR','currFeatures_curated','nestedFeature_struct');

%% Old code:

% Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
% % currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) str2num(x{2}),currFeatures_curated);
% currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);
% Features_to_process = setdiff(Features_to_process,currFeatures_curated);
% % if ~isempty(Featurefiles_dir) && isempty(Featurefiles_curated_dir)
% if ~isempty(Featurefiles_dir) && ~isempty(Features_to_process)
%     Featurefiles_names = {Featurefiles_dir(:).name}; curr_file_order = cell2mat(cellfun(@(x) str2num(x((strfind(x,'h') + 1):(strfind(x,'.') - 1))),Featurefiles_names,'UniformOutput',0));
%     [~, curr_file_order_sorted_idx] = sort(curr_file_order);
%     parfor i = 1:length(Features_to_process) % This is the PARFOR 
%         Feature = []; Feature_labels = [];
%         for j = 1:length(curr_file_order_sorted_idx)
%             
%             curr_Epoch = curr_file_order_sorted_idx(j);
%             disp(['Running Feature ' Features_to_process{i} ' Epoch ' num2str(curr_Epoch)]);
%             curr_file = load([Featurefiles_directory filesep Featurefiles_dir(curr_Epoch).name]);
%             
%             % Compute derivative features from original features if not included in saved file:
%             if ~isfield(curr_file,Features_to_process{i})
%                 if strcmp(Features_to_process{i},'CFC_SI_mag') || strcmp(Features_to_process{i},'CFC_SI_theta')
%                     for m = 1:length(curr_file.CFC_SI{1}) % Number of Windows
%                         for n = 1:size(curr_file.CFC_SI{1}{1},2) % Number of Frequency Windows - columns
%                             for p = 1:size(curr_file.CFC_SI{1}{1},1) % Number of Frequency Windows 2 - rows
%                                 if ~isempty(curr_file.CFC_SI{1}{m}{p,n})
%                                     num_chan = size(curr_file.CFC_SI{1}{m}{p,n},1);
%                                     for ch1 = 1:num_chan                                        
%                                         curr_file.CFC_SI_mag{1}{m}{p,n}(ch1,:) = abs(curr_file.CFC_SI{1}{m}{p,n}(ch1,:));
%                                         curr_file.CFC_SI_theta{1}{m}{p,n}(ch1,:) = angle(curr_file.CFC_SI{1}{m}{p,n}(ch1,:));
%                                     end
%                                 elseif p == size(curr_file.CFC_SI{1}{1},1)
%                                     curr_file.CFC_SI_mag{1}{m}{p,n} = [];
%                                 end
%                             end
%                         end
%                     end
%                 end
%             end
% 
%             
%             if isfield(curr_file,'analyzedData')
%                 curr_file_cell = struct2cell(curr_file.analyzedData); curr_file_cell_var = fields(curr_file.analyzedData);
%             else
%                 curr_file_cell = struct2cell(curr_file); curr_file_cell_var = fields(curr_file);
%             end
%             curr_feature_idx = cell2mat(cellfun(@(x) strcmp(x,Features_to_process{i}), curr_file_cell_var,'UniformOutput',0));
%             curr_Feature = curr_file_cell{curr_feature_idx};
%             
%             % Give feature output as cell or as a matrix:
%             if cell_output
%                 Feature = [Feature; curr_Feature];
%             else
%                 if size(curr_Feature{1}{1},1) == 1 % Single Frequency Computation
%                     curr_Feature_curated = NaN([length(curr_Feature{1}),length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
%                     curr_feature_label_curated = cell([length(curr_Feature{1}),length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
%                     
%                 else % Between Frequency Computation
%                     feat_size = [];
%                     for k = 1:length(size(curr_Feature{1}{1}{1}))
%                         feat_size{k} = max(max(cell2mat(cellfun(@(x) size(x,k),curr_Feature{1}{1},'UniformOutput',0))));                        
%                     end
%                     feat_size = cell2mat(feat_size);
%                     curr_Feature_curated = NaN([length(curr_Feature{1}),size(curr_Feature{1}{1}),feat_size]);
%                     curr_Feature_label_curated = NaN([length(curr_Feature{1}),size(curr_Feature{1}{1}),feat_size]);
%                 end
%                     
%                 for m = 1:length(curr_Feature{1}) % Number of Windows
%                     curr_feature_label = '';
%                     curr_feature_label = [curr_feature_label 'W' num2str(m)];
%                     for n = 1:size(curr_Feature{1}{1},2) % Number of Frequency Windows
%                         curr_feature_label = [curr_feature_label 'F' num2str(n)];
%                         if size(curr_Feature{1}{m},1) == 1 % Single Frequency Computation
%                             curr_Feature_curated(m,n,:,:) = curr_Feature{1}{m}{n};
%                         else % Between Frequency Computation
%                             for p = 1:size(curr_Feature{1}{1},1)
%                                 if ~isempty(curr_Feature{1}{m}{p,n}) curr_Feature_curated(m,p,n,:,:) = curr_Feature{1}{m}{p,n}; end    % TO FIX - need to add feat_size to this to make it compatible with features that are not a single value per pair of electrodes 
%                             end
%                         end
%                     end
%                 end
%                 Feature{j} = curr_Feature_curated(:);
%             end
%         end 
%         analyzedData = []; analyzedData.Feature = Feature; analyzedData.Feature_size = size(curr_Feature_curated); analyzedData.curr_Feature_curated = curr_Feature_curated;
%         parsave_struct([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}], analyzedData, 0)
%         % parsave([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}],Feature,Feature_size);
%     end
%     features_curated = true;
% elseif isempty(Featurefiles_dir)
%     disp('\n Features not computed yet \n');
% else
%     disp('\n Features already curated \n');
%     features_curated = true;
% end