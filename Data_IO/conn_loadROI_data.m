% Loads the ICA data from CONN processed ICA results
function output_data = conn_loadROI_data(CONN_cfg,scan_parameters)

% CONNresults_folder,CONN_data_type,CONN_project_name,net_to_analyze,use_All_cond

%% Get the number of participants, sessions and scans:
load([CONN_cfg.CONNresults_folder filesep CONN_cfg.CONN_project_name '.mat']);
file_suffix = 'tfunc'; % The suffix of the file that was processed in CONN - to decide if task based or rest
total_num_scans = cellfun(@(x) sum(cell2mat(x)),CONN_x.Setup.nscans);
num_subs = length(CONN_x.Setup.conditions.values);

for isub = 1:num_subs
    TR = CONN_x.Setup.RT(min(numel(CONN_x.Setup.RT),isub));
    [num_sess,num_sess_idx] = unique(cellfun(@(x)x{1},CONN_x.Setup.functional{isub},'un',0));
    [num_sess_idx,sort_i] = sort(num_sess_idx); num_sess = num_sess(sort_i);
    
    % Find relevant sessions:
    num_sess_curr = cellfun(@(x) strsplit(x,{'/','\'}),num_sess,'un',0); num_sess_curr = cellfun(@(x)x{end},num_sess_curr,'un',0);
    num_sess_curr_select = cellfun(@(x) ~isempty(strfind(x,file_suffix)),num_sess_curr);
    num_sess_idx_select = num_sess_idx(num_sess_curr_select);
    
    % Cumulatively add the sizes of the relevant sessions and save the start and end indicies of the sessions - 
    num_scans = cell2mat(CONN_x.Setup.nscans{isub});
    num_scans_cum = cumsum(num_scans);
    num_scans_select = num_scans(num_sess_idx_select);
    data_endIDX{isub} = num_scans_cum(num_sess_idx_select);
    data_startIDX{isub} = data_endIDX{isub} - num_scans_select + 1;    
end

%% Load the ROI data:
CONNdata_timeseries = [];
CONNdata_timeseries.conditions = CONN_x.Setup.conditions.allnames;
CONNdata_timeseries.data = cell(1,num_subs);
CONNdata_timeseries.data = cellfun(@(x)cell(1,length(CONNdata_timeseries.conditions)),CONNdata_timeseries.data,'un',0);

% Get the node label data:
full_CONN_node_labels = load([CONN_cfg.CONNresults_folder filesep 'ROI_Subject' sprintf('%03d',1) '_Condition' num2str(1,'%03d') '.mat'],'names'); full_CONN_node_labels = full_CONN_node_labels.names;
full_CONN_node_labels_cell = cellfun(@(y)cellfun(@(x)contains(x,y),full_CONN_node_labels),CONN_cfg.ROIs_toUse,'un',0);
ROIs_toUse = cellfun(@(x)find(x),full_CONN_node_labels_cell,'un',0);
CONN_node_labels = CONN_cfg.ROIs_toUse;

for i = 1:num_subs  
    for m = 1:length(CONNdata_timeseries.conditions)
        curr_data = load([CONN_cfg.CONNresults_folder filesep 'ROI_Subject' sprintf('%03d',i) '_Condition' num2str(m,'%03d') '.mat'],'data');
        time_length = size(curr_data.data{1},1);
        data_curated = zeros(time_length,length(CONN_cfg.ROIs_toUse));       
        for j = 1:size(data_curated,2)
            temp_X = cell2mat(curr_data.data(ROIs_toUse{j}));
            if size(temp_X,2) > 1 temp_X = mean(temp_X,2); end
            if isempty(temp_X) temp_X = nan(time_length,1); end
            if (length(temp_X) ~= time_length) pad_amt = time_length - length(temp_X); temp_X = [temp_X; zeros(pad_amt,1)]; end
            data_curated(:,j) = temp_X;
        end
        
        if CONN_cfg.rescale % Rescale between 0 and 1 if selected:
            CONNdata_timeseries.data{i}{m} = rescale(data_curated);
        else
            CONNdata_timeseries.data{i}{m} = data_curated;
        end
    end
end

%% Isolate the network activity of the networks of interest:
if CONN_cfg.use_All_cond
    cond_All = find(cellfun(@(x)~isempty(x),strfind(CONNdata_timeseries.conditions,'All')));
    if isempty(cond_All)
        CONN_cfg.use_All_cond = 0; % "All" condition does not exist in this dataset
    end
end

net_idx = cellfun(@(x)(cellfun(@(y)~isempty(strfind(y,x)),CONN_node_labels,'UniformOutput',0)),CONN_cfg.net_to_analyze,'UniformOutput',0);
net_idx = cellfun(@(x) find(cell2mat(x)),net_idx,'UniformOutput',0);
for i = 1:length(net_idx)
    for j = 1:length(net_idx{i})
        net_idx_label{i}{j} = CONN_node_labels{net_idx{i}(j)};
    end
end

%% Use L1/L2-Normalization to normalize the activity data:
% CONN_cfg.p_norm = 1; % can switch from L1 to L2 norm
% only run this if this is non-zero
CONN_cfg.conditions_to_include = [1 2]; % The condition indices to sum up in the norm
if CONN_cfg.use_All_cond
    if CONN_cfg.p_norm
        % data_norm = cellfun(@(data_raw) data_raw{cond_All}./(repmat(sum(abs(data_raw{cond_All}),2),1,size(data_raw{cond_All},2))),data,'UniformOutput',0); % Need to double check this - Normalizing across all components
        temp_data = cellfun(@(data_raw) mat2cell(data_raw{cond_All},size(data_raw{cond_All},1), ones(1,size(data_raw{cond_All},2))),CONNdata_timeseries.data,'un',0);
        temp_norm = cellfun(@(x) cellfun(@(y) norm(y,CONN_cfg.p_norm),x),temp_data,'un',0);
        data_norm = cellfun(@(data_raw,norm) data_raw{cond_All}./(repmat(norm,size(data_raw{cond_All},1),1)),CONNdata_timeseries.data,temp_norm,'UniformOutput',0);
    else
        data_norm = cellfun(@(x)x{cond_All},CONNdata_timeseries.data,'un',0);
    end
else
    % temp_data = cellfun(@(x) x{1} + x{2},data,'un',0);
    temp_data = cellfun(@(x) sum(cat(3,x{CONN_cfg.conditions_to_include}),3),CONNdata_timeseries.data,'un',0);
    if CONN_cfg.p_norm
        temp_data = cellfun(@(data_raw) mat2cell(data_raw,size(data_raw,1), ones(1,size(data_raw,2))),temp_data,'un',0);
        temp_norm = cellfun(@(x) cellfun(@(y) norm(y,CONN_cfg.p_norm),x),temp_data,'un',0); temp_data = cellfun(@(x) sum(cat(3,x{CONN_cfg.conditions_to_include}),3),CONNdata_timeseries.data,'un',0);
        data_norm = cellfun(@(data_raw,norm) data_raw./(repmat(norm,size(data_raw,1),1)),temp_data,temp_norm,'UniformOutput',0);
        % data_norm = cellfun(@(data_raw) data_raw./(repmat(sum(abs(data_raw),2),1,size(data_raw,2))),temp_data,'UniformOutput',0); % Need to double check this - Normalizing across all components
    else
        data_norm = temp_data;
    end
end

%% Compute the fMRI Labels of the selected components:

fMRI_labels_name = CONN_node_labels';

TR_window_step = CONN_cfg.window_step/scan_parameters.TR; % The window_step in terms of the TR
TR_window_length = CONN_cfg.window_length/scan_parameters.TR; % The window_length in terms of the TR

% CONN_cfg.threshold = 0.3;

for i = 1:length(data_startIDX)
    
    fMRI_labels{i} = arrayfun(@(y,z)data_norm{i}(y:z,:),data_startIDX{i},data_endIDX{i},'un',0); % fMRI_labels is size (num_sub X num_sessions)
    fMRI_labels_thresh{i} = cellfun(@(x)abs(x) >= CONN_cfg.threshold,fMRI_labels{i},'un',0); % Threshold the label data to make it binary 
    
    % Create the fMRI windows:
    vol_latencies{i} = cellfun(@(x)size(x,1),fMRI_labels{i});
    for j = 1:length(vol_latencies{i})
        [vol_start_idx{i}{j}, vol_end_idx{i}{j}] = create_windows(vol_latencies{i}(j), TR_window_step, TR_window_length); % Compute the start and end indicies in terms of MR volumes
        fMRI_labels_window{i}{j} = arrayfun(@(x,y)fMRI_labels{i}{j}(x:y,:),vol_start_idx{i}{j},vol_end_idx{i}{j},'un',0);
        
        fMRI_labels_window_thresh{i}{j} = cellfun(@(x)abs(x) >= CONN_cfg.threshold,fMRI_labels_window{i}{j},'un',0); % Threshold the label data to make it binary
               
        % Save selected components only:
        fMRI_labels_selected_window{i}{j} = cell(size(fMRI_labels_window{i}{j})); fMRI_labels_selected{i}{j} = []; fMRI_labels_name_selected = [];
        for k = 1:length(CONN_cfg.net_to_analyze)
            switch CONN_cfg.class_types
                case 'subnetworks'
                    fMRI_labels_selected{i}{j} = cat(2,fMRI_labels_selected{i}{j},fMRI_labels{i}{j}(:,net_idx{k})); % fMRI_labels_selected is size (num_sub X num_sessions X net_to_analyze)
                    fMRI_labels_selected_window{i}{j} = cellfun(@(x,y)cat(2,y,x(:,net_idx{k})),fMRI_labels_window{i}{j},fMRI_labels_selected_window{i}{j},'un',0); % fMRI_labels_selected_window is size (num_sub X num_sessions X net_to_analyze)
                    fMRI_labels_name_selected = cat(2,fMRI_labels_name_selected,net_idx_label{k});
                    
                case 'networks'
                    fMRI_labels_selected{i}{j} = cat(2,fMRI_labels_selected{i}{j},mean(fMRI_labels{i}{j}(:,net_idx{k}),2)); % fMRI_labels_selected is size (num_sub X num_sessions X net_to_analyze)
                    fMRI_labels_selected_window{i}{j} = cellfun(@(x,y)cat(2,y,mean(x(:,net_idx{k}),2)),fMRI_labels_window{i}{j},fMRI_labels_selected_window{i}{j},'un',0); % fMRI_labels_selected_window is size (num_sub X num_sessions X net_to_analyze)
                    fMRI_labels_name_selected{k} = CONN_cfg.net_to_analyze{k};
                      
            end            
        end
        
        % Find average activity for each network, for each window:
        fMRI_labels_selected_window_avg{i}{j} = cellfun(@(x)mean(x,1),fMRI_labels_selected_window{i}{j},'un',0);
        
        % Threshold the output to get binary labels for classification:
        fMRI_labels_selected_thresh{i}{j} = [];
        if CONN_cfg.multilabel
            for k = 1:size(fMRI_labels_selected{i}{j},1)
                fMRI_labels_selected_thresh{i}{j}{k} = find(fMRI_labels_selected{i}{j}(k,:) >= CONN_cfg.threshold); % Threshold the label data to make it binary
                fMRI_labels_selected_thresh{i}{j}{k}(isempty(fMRI_labels_selected_thresh{i}{j}{k})) = 0;
            end
            
            for m = 1:size(fMRI_labels_selected_window{i}{j},2)
                fMRI_labels_selected_window_thresh{i}{j}{m} = [];
                for k = 1:size(fMRI_labels_selected_window{i}{j}{m},1)
                    fMRI_labels_selected_window_thresh{i}{j}{m}{k} = find(fMRI_labels_selected_window{i}{j}{m}(k,:) >= CONN_cfg.threshold); % Threshold the label data to make it binary
                    fMRI_labels_selected_window_thresh{i}{j}{m}{k}(isempty(fMRI_labels_selected_window_thresh{i}{j}{m}{k})) = 0;
                end

                fMRI_labels_selected_window_avg_thresh{i}{j}{m} = find(fMRI_labels_selected_window_avg{i}{j}{m} >= CONN_cfg.threshold); % Threshold the label data to make it binary
                if isempty(fMRI_labels_selected_window_avg_thresh{i}{j}{m}) fMRI_labels_selected_window_avg_thresh{i}{j}{m} = 0; end
            end            
             
            
%             fMRI_labels_selected_thresh{i}{j} = find(abs(fMRI_labels_selected{i}{j}) >= CONN_cfg.threshold); % Threshold the label data to make it binary
%             fMRI_labels_selected_window_thresh{i}{j} = cellfun(@(x) find(abs(x) >= CONN_cfg.threshold),fMRI_labels_selected_window{i}{j},'un',0); % Threshold the label data to make it binary
        else
            [max_thresh,max_thresh_idx] = max(fMRI_labels_selected{i}{j},[],2);
            fMRI_labels_selected_thresh{i}{j} = zeros(size(max_thresh_idx));
            fMRI_labels_selected_thresh{i}{j}(max_thresh >= CONN_cfg.threshold) = max_thresh_idx(max_thresh >= CONN_cfg.threshold); % Threshold the label data to make it binary          

            [max_thresh,max_thresh_idx] = cellfun(@(x)max(x,[],2),fMRI_labels_selected_window{i}{j},'un',0);
            [max_thresh_avg,max_thresh_avg_idx] = cellfun(@(x)max(x),fMRI_labels_selected_window_avg{i}{j},'un',0);

            fMRI_labels_selected_window_thresh{i}{j} = cellfun(@(x,y)zeros(size(y)),fMRI_labels_selected_window{i}{j},max_thresh_idx,'un',0); % Threshold the label data to make it binary
            fMRI_labels_selected_window_avg_thresh{i}{j} = cellfun(@(x)zeros(1,1),fMRI_labels_selected_window_avg{i}{j},'un',0); % Threshold the label data to make it binary

            for k = 1:length(max_thresh) 
                fMRI_labels_selected_window_thresh{i}{j}{k}(max_thresh{k} >= CONN_cfg.threshold) = max_thresh_idx{k}(max_thresh{k} >= CONN_cfg.threshold);
                
                if (max_thresh_avg{k} >= CONN_cfg.threshold)
                    fMRI_labels_selected_window_avg_thresh{i}{j}{k} = max_thresh_avg_idx{k};
                end
                
            end
        end
    end
    
% fMRI_labels_selected{i}{j}{k} = fMRI_labels{i}{j}(:,net_idx{k}); % fMRI_labels_selected is size (num_sub X num_sessions X net_to_analyze)
% fMRI_labels_selected_window{i}{j}{k} = cellfun(@(x)x(:,net_idx{k}),fMRI_labels_window{i}{j},'un',0); % fMRI_labels_selected_window is size (num_sub X num_sessions X net_to_analyze)
% 
% fMRI_labels_selected_thresh{i}{j}{k} = abs(fMRI_labels_selected{i}{j}{k}) >= CONN_cfg.threshold; % Threshold the label data to make it binary
% fMRI_labels_selected_window_thresh{i}{j}{k} = cellfun(@(x)abs(x) >= CONN_cfg.threshold,fMRI_labels_selected_window{i}{j}{k},'un',0); % Threshold the label data to make it binary


end

%% Write output structure:
output_data = [];
output_data.fMRI_labels = fMRI_labels;
output_data.fMRI_labels_thresh = fMRI_labels_thresh;
output_data.fMRI_labels_window = fMRI_labels_window;
output_data.fMRI_labels_window_thresh = fMRI_labels_window_thresh;
output_data.fMRI_labels_selected = fMRI_labels_selected;
output_data.fMRI_labels_selected_window = fMRI_labels_selected_window;
output_data.fMRI_labels_selected_thresh = fMRI_labels_selected_thresh;
output_data.fMRI_labels_selected_window_thresh = fMRI_labels_selected_window_thresh;
output_data.fMRI_labels_selected_window_avg = fMRI_labels_selected_window_avg;
output_data.fMRI_labels_selected_window_avg_thresh = fMRI_labels_selected_window_avg_thresh;
output_data.fMRI_labels_name = fMRI_labels_name;
output_data.fMRI_labels_name_selected = fMRI_labels_name_selected;

%% Code from iterate_create_EEGfMRI_dataset2

% % Check if processed fMRI data exists:
% fMRI_subjects = unique(IDX_subject);
% fMRI_sessions = unique(IDX_session);
% curr_fMRI_dataset = zeros(max(fMRI_subjects),max(fMRI_sessions));
% for i = 1:max(fMRI_subjects) % fMRI Subjects
%     i_sess_max = max(IDX_session(IDX_subject == i)); % Total number of sessions for this fMRI subject
%     for j = 1:i_sess_max % fMRI Sessions
%         fMRI_anat_file = CONN_x.Setup.structural{i}{j}{1};
%         [~,name,~] = fileparts(fMRI_anat_file);
%         curr_fMRI_dataset(i,j) = ~isempty(strfind(name,num2str(curr_exam)));
%     end
% end
% clear i j i_sess_max fMRI_anat_file name
% fMRI_present = sum(curr_fMRI_dataset(:)) > 0;
% 
% if fMRI_present
%     % Load the processed fMRI data for the selected ROIs:
%     [fMRI_subj_idx,fMRI_sess_idx] = find(curr_fMRI_dataset);
%     fMRI_data = Y((IDX_session == fMRI_sess_idx) & (IDX_subject == fMRI_subj_idx),~boolean(ROIs_toExtract)); fMRI_data_name = ROInames(~boolean(ROIs_toExtract));
%     
%     switch fMRI_label_mode
%         case 'ROI'
%             fMRI_labels = Y((IDX_session == fMRI_sess_idx) & (IDX_subject == fMRI_subj_idx),boolean(ROIs_toExtract)); fMRI_labels_name = ROInames(boolean(ROIs_toExtract));
%             % To select ROIs for tri-network model
%             select_vect = zeros(1,size(fMRI_labels,2)); % To select ROIs for tri-network model
%             select_vect(1) = 1; select_vect(4:9) = 1; select_vect(21:26) = 1; select_vect(29) = 1; select_vect(31) = 1; select_vect(33) = 1;
%             
%         case 'ICA'
%             subj_ICA = data{fMRI_subj_idx}{1};
%             sess_ICA_endidx =  fMRI_sess_idx*num_Temporal_pts; sess_ICA_startidx = sess_ICA_endidx - num_Temporal_pts + 1;
%             fMRI_labels = subj_ICA(sess_ICA_startidx:sess_ICA_endidx,:); fMRI_labels_name = ICA_comp_name';
%             % To select ROIs for tri-network model
%             select_vect = zeros(1,size(fMRI_labels,2));
%             select_vect(5:7) = 1; select_vect(9) = 1; select_vect(12) = 1;
%     end
%     % Save selected ROIs for tri-network model (CEN, DMN and Salience Network nodes):
%     fMRI_labels_selected = fMRI_labels(:,boolean(select_vect));
%     fMRI_labels_name_selected = fMRI_labels_name(boolean(select_vect));
%     
%     % Create the fMRI windows:
%     fMRI_data_window = zeros(total_len,fMRI_window,size(fMRI_data,2));
%     fMRI_labels_window = zeros(total_len,fMRI_window,size(fMRI_labels,2));
%     fMRI_labels_selected_window = zeros(total_len,fMRI_window,size(fMRI_labels_selected,2));
%     for i = 1:total_len
%         fMRI_data_window(i,:,:) = fMRI_data(fMRI_window_vect_startidx(i):fMRI_window_vect_endidx(i),:);
%         fMRI_labels_window(i,:,:) = fMRI_labels(fMRI_window_vect_startidx(i):fMRI_window_vect_endidx(i),:);
%         fMRI_labels_selected_window(i,:,:) = fMRI_labels_selected(fMRI_window_vect_startidx(i):fMRI_window_vect_endidx(i),:);
%     end
%     fMRI_labels_window = squeeze(mean(fMRI_labels_window,2));
%     fMRI_labels_selected_window = squeeze(mean(fMRI_labels_selected_window,2));
%     
%     % Use L1-Normalization to normalize the label data:
%     curr_norm = repmat(sum(abs(fMRI_labels),2),1,size(fMRI_labels,2));
%     fMRI_labels = fMRI_labels./curr_norm;
%     
%     curr_norm = repmat(sum(abs(fMRI_labels_selected),2),1,size(fMRI_labels_selected,2));
%     fMRI_labels_selected = fMRI_labels_selected./curr_norm;
%     
%     curr_norm = repmat(sum(abs(fMRI_labels_window),2),1,size(fMRI_labels_window,2));
%     fMRI_labels_window = fMRI_labels_window./curr_norm;
%     
%     curr_norm = repmat(sum(abs(fMRI_labels_selected_window),2),1,size(fMRI_labels_selected_window,2));
%     fMRI_labels_selected_window = fMRI_labels_selected_window./curr_norm;
%     
%     % Threshold the label data to make it binary:
%     threshold = 0.3;
%     fMRI_labels_thresh = abs(fMRI_labels) >= threshold;
%     fMRI_labels_selected_thresh = abs(fMRI_labels_selected) >= threshold;
%     fMRI_labels_window_thresh = abs(fMRI_labels_window) >= threshold;
%     fMRI_labels_selected_window_thresh = abs(fMRI_labels_selected_window) >= threshold;
% end