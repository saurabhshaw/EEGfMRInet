study_name = 'CompositeTask';
conn_project_name = 'conn_composite_task_fMRI';
%conn_project_name = 'conn_concuss_63_time1_HC_OBI';
conn_folder = 'Z:\Research_data\Analyzed_data\Composite_Task_EEG_fMRI';

nsub = 3;
nsess = 4;
v2v_analysis_type = 'V2V_02';

net_to_analyze = {'CEN', 'DMN', 'SN'}; % List all networks to analyze
use_All_cond = 0; % If this is 1, use the ICA timecourses from condition 'All'
task_epoch_delta = 15; % The number of MRI volumes before and after the task onset to epoch, i.e [-task_epoch_delta task_epoch_delta]
print_img = 0;

% ICA_component = 2;

[base_path_rc, base_path_rd] = setPaths();
addpath([base_path_rc filesep 'Toolboxes']);
addpath(genpath([base_path_rc filesep 'Toolboxes' filesep 'conn18b']));
addpath(genpath([base_path_rc filesep 'Toolboxes' filesep 'spm12']));
addpath(genpath([base_path_rc filesep 'Toolboxes' filesep 'mvgc_v1.0']));
addpath([base_path_rc filesep 'EEG_fMRI_Modelling' filesep 'Joint_Feature_Learning' filesep 'Utils']);

%% Load ICA CONN data (from conn_ica_explore.m):
% Load ICA data:
ICAresults_folder = [conn_folder filesep conn_project_name filesep 'results' filesep 'firstlevel' filesep v2v_analysis_type];
%ICAresults_folder = 'Z:\Research_data\Analyzed_data\Rachelle_data'; conn_folder = 'Z:\Research_data\Analyzed_data\Rachelle_data'; 
load([ICAresults_folder filesep 'ICA.Timeseries.mat']);
load([conn_folder filesep conn_project_name '.mat']);
fid = fopen([ICAresults_folder filesep 'ICA.ROIs.txt']); ICA_labels = textscan(fid,'%s'); ICA_labels = ICA_labels{1}; fclose(fid);

% Load Canonical Reference T1 atlas (from conn.m line 727 to 629 - conn18b)
filename = fullfile(fileparts(which('conn')),'utils','surf','referenceT1_trans.nii');
V = spm_vol(filename);
CONN_gui.refs.canonical = struct('filename',filename,'V',V,'data',spm_read_vols(V));

% Load ROI data atlas (from conn.m line 732 to 735 - conn18b)
filename = fullfile(fileparts(which('conn')),'rois','atlas.nii'); [filename_path,filename_name,filename_ext] = fileparts(filename);
V = spm_vol(filename);
CONN_gui.refs.rois = struct('filename',filename,'filenameshort',filename_name,'V',V,'data',spm_read_vols(V),'labels',{textread(fullfile(filename_path,[filename_name,'.txt']),'%s','delimiter','\n')});

% Load the ICA volumes:
vol = spm_vol([ICAresults_folder filesep 'ICA.ROIs.nii']);
B = spm_read_vols(vol); [maxB,imaxB] = max(B,[],4);
[x,y,z] = ndgrid(1:vol(1).dim(1),1:vol(1).dim(2),1:vol(1).dim(3));
xyz = vol(1).mat*[x(:) y(:) z(:) ones(numel(x),1)]';
imat = pinv(vol(1).mat);

%% Process ICA vol data (from conn_ica_explore.m):
volref = CONN_gui.refs.canonical.V;
Bref = reshape(spm_get_data(volref,pinv(volref.mat)*xyz),vol(1).dim(1:3)).^2;

TR = CONN_x.Setup.RT;
Ncomponents = size(B,4);
KURx = zeros(1,Ncomponents); SKWx = zeros(1,Ncomponents);
for n = 1:Ncomponents
    b = B(:,:,:,n);
    mask =~ isnan(b); mb = mean(b(mask)); sb = std(b(mask),0);
    KURx(n) = mean(((b(mask)-mb)/sb).^4);
    SKWx(n) = mean(((b(mask)-mb)/sb).^3);
end
CCx = nan(1,Ncomponents);
VARt = zeros(numel(data{1}),Ncomponents);
ZCt = zeros(numel(data{1}),Ncomponents);
validc = zeros(numel(data{1}),1);
for n1 = 1:numel(data) % subjects
    for n2 = 1:numel(data{n1}) % conditions
        if ~isempty(data{n1}{n2})
            mask = weights{n1,n2}{1}>0;
            b = data{n1}{n2}(mask,:)*sqrt(nnz(mask));
            remove = weights{n1,n2}{5}(mask)==1;
            VARt(n2,:) = VARt(n2,:)+std(b,0,1);
            mb = mean(b,1);
            b = conn_bsxfun(@minus,b,mb);
            b(remove,:) = 0;
            ZCt(n2,:) = ZCt(n2,:)+((mean((b(1:end-1,:)<=0&b(2:end,:)>0)|(b(1:end-1,:)>=0&b(2:end,:)<0),1))/TR(min(numel(TR),n1)))/2;
            validc(n2) = validc(n2)+1;
        end
    end
end
VARt = conn_bsxfun(@rdivide,VARt,validc);
ZCt = conn_bsxfun(@rdivide,ZCt,validc);

%% Read corrected timing data:
sub_info_table = readtable([base_path_rd filesep 'Subject_Info_' study_name '.xls']);
sub_foldername = cellfun(@(x,y)[erase(x,'-') '_' study_name '_EEGfMRI-' y],sub_info_table.Date,sub_info_table.SubjectID,'un',0);
sub_trial_data = cellfun(@(x,y) load([base_path_rd filesep x filesep y '_trial_data.mat'],'block_condition_onset_vect','block_condition_duration_vect'),sub_foldername,sub_info_table.SubjectID,'un',0);

% This is specific to my dataset:
max_trial_runs = 4;
sub_trial_data_relevant_idx = boolean([0 ones(1,length(sub_trial_data)-1)]); 
sub_trial_data_relevant = sub_trial_data(sub_trial_data_relevant_idx);
for i = 1:length(sub_trial_data_relevant)
    curr_onset = sub_trial_data_relevant{i}.block_condition_onset_vect;
    curr_durat = sub_trial_data_relevant{i}.block_condition_duration_vect;
    
    % Remove extra faulty trials (only at the beginning):
    if (length(curr_onset)>max_trial_runs) curr_onset(1) = []; curr_durat(1) = []; end
    
    % Add extra trials at the end if the number of trials is less than the max_trial_runs:
    if (length(curr_onset) < max_trial_runs)
        curr_onset = [curr_onset {cell(1,length(curr_onset{1}) + 1)}];
        curr_durat = [curr_durat {cell(1,length(curr_onset{1}) + 1)}];
    end
    
    for j = 1:length(curr_onset)
        % Add the "Switch" condition slot:
        curr_onset{j}{6} = [curr_onset{j}{4} curr_onset{j}{5}];
        curr_durat{j}{6} = [curr_durat{j}{4} curr_durat{j}{5}];
        
        % Add the "All" condition slot:
        curr_onset{j}{7} = 0;
        curr_durat{j}{7} = Inf;
    end
    
    % Add extra empty slot for "rest" session:
    curr_onset = [curr_onset {cell(1,length(curr_onset{j}))}];
    curr_durat = [curr_durat {cell(1,length(curr_onset{j}))}];

    sub_trial_data_relevant{i}.block_condition_onset_vect = curr_onset;
    sub_trial_data_relevant{i}.block_condition_duration_vect = curr_durat;
end

%% Isolate the time course for the conditions/tasks:
task_tc = cell(size(weights,1),size(weights,2),size(weights{1,1},2));
task_onset = cell(size(weights,1),size(weights,2));
task_duration = cell(size(weights,1),size(weights,2));
task_session = cell(size(weights,1),size(weights,2));
task_onset_corr = cell(size(weights,1),size(weights,2));
task_duration_corr = cell(size(weights,1),size(weights,2));
task_session_corr = cell(size(weights,1),size(weights,2));
task_tc_orig = cell(size(weights,1),size(weights,2));
total_num_scans = cellfun(@(x) sum(cell2mat(x)),CONN_x.Setup.nscans);
num_subs = length(CONN_x.Setup.conditions.values);
for isub = 1:num_subs
    TR = CONN_x.Setup.RT(min(numel(CONN_x.Setup.RT),isub));   
    num_cond = length(CONN_x.Setup.conditions.values{isub});
    for icond = 1:num_cond
        num_scans = cell2mat(CONN_x.Setup.nscans{isub});
        task_tc_orig{isub, icond} = zeros(1,max(total_num_scans));
        num_sess = length(CONN_x.Setup.conditions.values{isub}{icond});
        for isess = 1:num_sess
            
            % Calculated from the "weights" in ICA.timeseries.mat file
            task_tc{isub, icond, isess} = weights{isub,icond}{isess} ~= 0; 
            
            % Calculated from the values defined in Setup.conditions:
            onset = CONN_x.Setup.conditions.values{isub}{icond}{isess}{1}; 
            durat = CONN_x.Setup.conditions.values{isub}{icond}{isess}{2};
            durat(isinf(durat)) = num_scans(isess)*TR;
            
            if num_scans(isess)*TR >= max(onset)
                offset = sum(num_scans((1:length(num_scans)) < isess)) + 1;
                task_onset{isub, icond} = [task_onset{isub, icond} floor(onset./TR)+offset];
                task_duration{isub, icond} = [task_duration{isub, icond} ceil(durat./TR)];   
                task_session{isub, icond} = [task_session{isub, icond} repmat(isess,[1,length(onset)])];
            end
            
            % Calculated from the corrected values saved in PID_trial_data generated by running conn_addSubjects_EEGfMRI
            onset_corr = sub_trial_data_relevant{isub}.block_condition_onset_vect{isess}{icond}; 
            durat_corr = sub_trial_data_relevant{isub}.block_condition_duration_vect{isess}{icond};
            durat_corr(isinf(durat_corr)) = num_scans(isess)*TR;
            
            if num_scans(isess)*TR >= max(onset)
                offset_corr = sum(num_scans((1:length(num_scans)) < isess)) + 1;
                task_onset_corr{isub, icond} = [task_onset_corr{isub, icond} floor(onset_corr./TR)+offset_corr];
                task_duration_corr{isub, icond} = [task_duration_corr{isub, icond} ceil(durat_corr./TR)];   
                task_session_corr{isub, icond} = [task_session_corr{isub, icond} repmat(isess,[1,length(onset_corr)])];
            end
            
        end
        
        % Generate time courses:
        for i = 1:length(task_onset{isub, icond})
            start_idx = task_onset{isub, icond}(i);
            upper_limit = sum(num_scans((1:length(num_scans)) <= task_session{isub, icond}(i)));
            end_idx = min((start_idx + task_duration{isub, icond}(i)),upper_limit);
            task_tc_orig{isub, icond}(start_idx:end_idx) = 1;
        end
    end
end

curr_task_tc = task_tc(nsub,:,nsess); curr_task_tc_orig = task_tc_orig(nsub,:);
% ABM_tc = curr_task_tc{1}; ABM_tc_orig = curr_task_tc_orig{1};
% WM_tc = curr_task_tc{2}; WM_tc_orig = curr_task_tc_orig{2};
% figure; plot(ABM_tc*0.01); hold on; plot(WM_tc*0.01);
% figure; plot(ABM_tc_orig*0.01); hold on; plot(WM_tc_orig*0.01);

%% Isolate the network activity of the networks of interest:
if use_All_cond 
    cond_All = find(cellfun(@(x)~isempty(x),strfind(conditions,'All'))); 
    if isempty(cond_All)
        use_All_cond = 0; % "All" condition does not exist in this dataset
    end
end
net_idx = cellfun(@(x)(cellfun(@(y)~isempty(strfind(y,x)),ICA_labels,'UniformOutput',0)),net_to_analyze,'UniformOutput',0);
net_idx = cellfun(@(x) find(cell2mat(x)),net_idx,'UniformOutput',0); 
for i = 1:length(net_idx)
    for j = 1:length(net_idx{i})
        net_idx_label{i}{j} = ICA_labels{net_idx{i}(j)};    
    end
end

%% Use L1/L2-Normalization to normalize the activity data:
p_norm = 2; % can switch from L1 to L2 norm
conditions_to_include = [1 2]; % The condition indices to sum up in the norm
if use_All_cond 
    % data_norm = cellfun(@(data_raw) data_raw{cond_All}./(repmat(sum(abs(data_raw{cond_All}),2),1,size(data_raw{cond_All},2))),data,'UniformOutput',0); % Need to double check this - Normalizing across all components
    temp_data = cellfun(@(data_raw) mat2cell(data_raw{cond_All},size(data_raw{cond_All},1), ones(1,size(data_raw{cond_All},2))),data,'un',0);
    temp_norm = cellfun(@(x) cellfun(@(y) norm(y,p_norm),x),temp_data,'un',0);  
    data_norm = cellfun(@(data_raw,norm) data_raw{cond_All}./(repmat(norm,size(data_raw{cond_All},1),1)),data,temp_norm,'UniformOutput',0);
else
    % temp_data = cellfun(@(x) x{1} + x{2},data,'un',0);
    temp_data = cellfun(@(x) sum(cat(3,x{conditions_to_include}),3),data,'un',0); 
    temp_data = cellfun(@(data_raw) mat2cell(data_raw,size(data_raw,1), ones(1,size(data_raw,2))),temp_data,'un',0);
    temp_norm = cellfun(@(x) cellfun(@(y) norm(y,p_norm),x),temp_data,'un',0); temp_data = cellfun(@(x) sum(cat(3,x{conditions_to_include}),3),data,'un',0);
    data_norm = cellfun(@(data_raw,norm) data_raw./(repmat(norm,size(data_raw,1),1)),temp_data,temp_norm,'UniformOutput',0);
    % data_norm = cellfun(@(data_raw) data_raw./(repmat(sum(abs(data_raw),2),1,size(data_raw,2))),temp_data,'UniformOutput',0); % Need to double check this - Normalizing across all components
end

% data_norm = data_raw./(repmat(sum(abs(data_raw),1),size(data_raw,1),1)); 
% plot(sum(data_norm{nsub}(:,DMN_ICA_comp),2));

%% Extract Epoched Trials:
close all
condition_idx = 2; % Look at "conditions" variable to identify which condition to analyze:
curr_task_onset = task_onset_corr; % Can either be regular "task_onset" or the corrected version "task_onset_corr"
data_comb = data_norm; % Can be data_norm or this: cellfun(@(x) x{cond_All},data,'un',0)

trial_onsets_comb = curr_task_onset(:,condition_idx);
trial_tc = zeros(1,task_epoch_delta*2+1); trial_tc(task_epoch_delta:end) = 1;
x_data = (-task_epoch_delta:task_epoch_delta)*TR;

start_idx_comb = cellfun(@(x) x - task_epoch_delta, trial_onsets_comb,'un',0);
end_idx_comb = cellfun(@(x) x + task_epoch_delta, trial_onsets_comb,'un',0); 

trial_comb = []; subject_trial_idx = [];
for i = 1:length(trial_onsets_comb) 
    idx_remove = start_idx_comb{i} <= 0; start_idx_comb{i}(idx_remove) = []; end_idx_comb{i}(idx_remove) = []; trial_onsets_comb{i}(idx_remove) = []; % Remove trials that start before the beginning
    idx_remove = end_idx_comb{i} > total_num_scans(i);start_idx_comb{i}(idx_remove) = []; end_idx_comb{i}(idx_remove) = []; trial_onsets_comb{i}(idx_remove) = []; % Remove trials that end after the last data point
    
    for j = 1:length(start_idx_comb{i})
        % trial_comb = cat(3,trial_comb, data_comb{i}(start_idx_comb{i}(j):end_idx_comb{i}(j),:));
        trial_comb = [trial_comb; {data_comb{i}(start_idx_comb{i}(j):end_idx_comb{i}(j),:)}];
        subject_trial_idx = [subject_trial_idx; i];
    end        
end

% get_trials = @(data, start_idx_arr, end_idx_arr) cellfun(@(start_idx,end_idx) data(start_idx:end_idx,:),num2cell(start_idx_arr), num2cell(end_idx_arr),'un',0);
% trial_comb = cellfun(@(x,y,z) get_trials(x,y,z),data_comb',start_idx_comb,end_idx_comb,'un',0);

%% Average within Subjects:
% nsub = 2;
% trial_avg = cell(1,length(data)); trial_std = cell(1,length(data));
% for i = 1:length(data)
%     trial_avg{i} = mean(cat(3,trial_comb{subject_trial_idx == i}),3); trial_std{i} = std(cat(3,trial_comb{subject_trial_idx == i}),[],3)./sqrt(sum(subject_trial_idx == i));    
% end
% 
% % Plot the figure:
% plot_errdisp = 'area'; % Can be 'bar' or 'area'
% load('color_order_shades.mat'); color_order_idx = [1 2 4];
% % plot_colorord = get(gca,'ColorOrder');
% plot_colorord = cell2mat(color_order(:,3)); % Index 3 is the same as the standard Matlab colors
% title_text = ['Within Subject Onset of ' conditions{condition_idx} ' Trials'];
% h = plot_network_timecourse(x_data,net_to_analyze,title_text,net_idx,trial_avg{nsub},trial_std{nsub},plot_colorord(color_order_idx,:),plot_errdisp);

%% Average across Subjects:
trial_avg_comb = mean(cat(3,trial_comb{:}),3); trial_ste_comb = std(cat(3,trial_comb{:}),[],3)./sqrt(length(trial_comb));

% Plot the figure:
plot_errdisp = 'area'; % Can be 'bar' or 'area'
load('color_order_shades.mat'); color_order_idx = [1 2 4];
plot_colorord = cell2mat(color_order(:,3)); % Index 3 is the same as the standard Matlab colors

% Plot the figure:
x_data = (-task_epoch_delta:task_epoch_delta)*TR; 
title_text = ['Onset of ' conditions{condition_idx} ' Trials'];
h = plot_network_timecourse(x_data,net_to_analyze,title_text,net_idx,trial_avg_comb,trial_ste_comb,plot_colorord(color_order_idx,:),plot_errdisp);
if print_img filename=regexprep(title_text, '_', ''); filename=regexprep(filename, ' ', '_'); print('-djpeg','-r500',[conn_folder filesep 'Images' filesep filename '.jpg']); end

%% Compute time-coherence between pairs of network activations:
base_net = 3; % Index of the net_to_analyze network against which all coherence values will be computed
corr_window = 5; % Window size to compute the correlation
interp_factor = 2;

trial_base_net = cellfun(@(trial_data)sum(trial_data(:,net_idx{base_net}),2),trial_comb,'un',0);
trial_corr_comb = cell(size(trial_comb,1),length(net_to_analyze));
trial_corr_avg_comb = cell(1,length(net_to_analyze)); trial_corr_ste_comb = cell(1,length(net_to_analyze));
trial_avg_comb_corr = cell(1,length(net_to_analyze)); trial_ste_comb_corr = cell(1,length(net_to_analyze));
for i = 1:length(net_to_analyze)
    curr_net_idx = net_idx{i};
    parfor j = 1:length(trial_comb)
        %trial_coherence_comb{j,i} = mscohere(sum(trial_comb{j}(:,curr_net_idx),2),trial_base_net{j});    
        %trial_corr_comb{j,i} = xcorr(sum(trial_comb{j}(:,curr_net_idx),2),trial_base_net{j},'normalized');
        A = interp(sum(trial_comb{j}(:,curr_net_idx),2),interp_factor); B = interp(trial_base_net{j},interp_factor);
        if mod(length(A),corr_window) A = A(1:end-mod(length(A),corr_window)); end
        if mod(length(B),corr_window) B = B(1:end-mod(length(B),corr_window)); end
        trial_corr_comb{j,i} = diag(corr(reshape(A,corr_window,[]),reshape(B,corr_window,[])));
    end
    % Average across Subjects:
    trial_corr_avg_comb{i} = mean(cat(3,trial_corr_comb{:,i}),3); trial_corr_ste_comb{i} = std(cat(3,trial_corr_comb{:,i}),[],3)./sqrt(size(trial_corr_comb,1));
    
    % Compute correlation on pre-averaged trials from previous section:
    A = interp(sum(trial_avg_comb(:,net_idx{i}),2),interp_factor); B = interp(sum(trial_avg_comb(:,net_idx{base_net}),2),interp_factor);
    if mod(length(A),corr_window) A = A(1:end-mod(length(A),corr_window)); end
    if mod(length(B),corr_window) B = B(1:end-mod(length(B),corr_window)); end
    trial_avg_comb_corr{i} = diag(corr(reshape(A,corr_window,[]),reshape(B,corr_window,[]))); trial_ste_comb_corr{i} = zeros(size(trial_avg_comb_corr{i}));
    
    % Plot the figure:
    % x_data = 1:length(trial_corr_avg_comb{i});
    x_data = linspace(-task_epoch_delta,task_epoch_delta,length(trial_corr_avg_comb{i}))*TR;
    title_text = ['Onset of ' conditions{condition_idx} ' Trials - Correlation ' net_to_analyze{i} ' vs ' net_to_analyze{base_net}];
    % h = plot_network_timecourse(x_data,{net_to_analyze{i}},title_text,{[1]},trial_corr_avg_comb{i},trial_corr_ste_comb{i},plot_errdisp);
    h = plot_network_timecourse(x_data,{net_to_analyze{i}},['Pre-avg ' title_text],{[1]},trial_avg_comb_corr{i},trial_ste_comb_corr{i},plot_colorord(1,:),plot_errdisp);
    if print_img filename=regexprep(title_text, '_', ''); filename=regexprep(filename, ' ', '_'); print('-djpeg','-r500',[conn_folder filesep 'Images' filesep filename '.jpg']); end
end

%% Plot all of the sub-networks individually:
% load('color_order_shades.mat'); % color_offset = 2;
color_order_idx = [1 2 4]; subcolor_order_idx = [1 5];
y_lim_value = [-0.01 0.01];
trial_avg_comb = mean(cat(3,trial_comb{:}),3); trial_ste_comb = std(cat(3,trial_comb{:}),[],3)./sqrt(length(trial_comb));
unique_subnets = unique([net_idx{:}]); % net_idx_done = zeros(length(net_idx),1);
colorord_cum = [];
for i = 1:length(net_idx)
    %for j = 1:length(net_idx{i})
        % Plot the figure:
        x_data = (-task_epoch_delta:task_epoch_delta)*TR;
        title_text = ['Onset of ' conditions{condition_idx} ' Trials-' net_to_analyze{i}]; title_text = regexprep(title_text, '_', '');
        % h = plot_network_timecourse(x_data,{ICA_labels{unique_subnets(i)}},title_text,{[1]},trial_avg_comb(:,unique_subnets(i)),trial_ste_comb(:,unique_subnets(i)),plot_colorord(i,:),plot_errdisp);
        % curr_net_idx = find(cellfun(@(x) ismember(net_idx{i}(j),x),net_idx)); net_idx_done(curr_net_idx) = net_idx_done(curr_net_idx) + 1;
        subplot(length(net_idx),1,i);
        curr_net_idx = mat2cell(net_idx{i},ones(size(net_idx{i})));
        curr_colorord = cell2mat({color_order{color_order_idx(i),subcolor_order_idx}}'); colorord_cum = cat(1,colorord_cum,curr_colorord);
        h = plot_network_timecourse_subplot(x_data,{ICA_labels{net_idx{i}}},title_text,curr_net_idx,trial_avg_comb,trial_ste_comb,curr_colorord,plot_errdisp);
        ylim(y_lim_value); line([0 0],ylim,'Color','k');
        % if print_img filename=regexprep(title_text, ' ', '_'); print('-djpeg','-r500',[conn_folder filesep 'Images' filesep filename '.jpg']); end
    %end
end
xlabel('Time(s)');
if print_img filename=regexprep(['Onset of ' conditions{condition_idx} ' Trials-SubNetworks'], ' ', '_'); print('-djpeg','-r500',[conn_folder filesep 'Images' filesep filename '.jpg']); end

%% Compute MVGC
mvgc_window_length = 10; mvgc_window_step = 1;
mvgc_demo_ICA

% Run Non-Stationary MVGC script:
X = data_curated;
mvgc_demo_nonstationary_ICA
if print_img filename=regexprep(['Onset of ' conditions{condition_idx} ' Trials-MVGC-timePerm'], ' ', '_'); print('-djpeg','-r600',[conn_folder filesep 'Images' filesep filename '.jpg']); end

%% Plot mvgc Graphs:
% A = full_mvgc_time.*full_mvgc_sig; node_names = data_curated_labels; node_colors = colorord_cum;
% [G,h,hfig] = plot_digraph(A,node_names,node_colors);
% title(['Onset of ' conditions{condition_idx} ' Trials-MVGC']);
% if print_img filename=regexprep(['Onset of ' conditions{condition_idx} ' Trials-MVGC'], ' ', '_'); print('-djpeg','-r600',[conn_folder filesep 'Images' filesep filename '.jpg']); end
node_colors = colorord_cum;
node_names = data_curated_labels; start_timeIDX = 10; 
subplotgrid = [4 4]; % Was [5 5]

% Make sure that the color order follows the same order as data_curated_labels
data_curated_idx = cat(1,net_idx{:});
new_net_idx = net_idx;
for i = 1:length(data_curated_idx)
    curr_net_idx = cellfun(@(x) find(x==data_curated_idx(i)),net_idx,'un',0);
    curr_cell_idx = cellfun(@(x) ~isempty(x),curr_net_idx);
    new_net_idx{curr_cell_idx}(curr_net_idx{curr_cell_idx}) = i;
end

hfig_cell = cell(length(windowed_mvgc_time),1); h_cell = cell(length(windowed_mvgc_time),1); subplot_cell = cell(length(windowed_mvgc_time),1); 
for i = start_timeIDX:length(windowed_mvgc_time)
    A = windowed_mvgc_time_NST{i}.*windowed_mvgc_sig_NST{i}; A(A < Fc) = 0;
    
    %[windowed_G{i},h_cell{i},hfig_cell{i}] = plot_digraph(A',node_names,node_colors);
     [windowed_G{i},h_cell{i},hfig_cell{i}] = plot_digraph_split(A',node_names,node_colors,new_net_idx);

    % title(['Onset of ' conditions{condition_idx} ' Trials-MVGC']);
    % if print_img filename=regexprep(['Onset of ' conditions{condition_idx} ' Trials-MVGC-time' sprintf('%02d',i)], ' ', '_'); print('-djpeg','-r600',[conn_folder filesep 'Images' filesep filename '.jpg']); end  
end

% Combine the individual images into a montage:
h_composite = figure('units','normalized','outerposition',[0 0 1 1]); subplot_offset = 3;
for j = start_timeIDX:length(windowed_mvgc_time)
    subplot_cell{j} = subplot(subplotgrid(1), subplotgrid(2), j - start_timeIDX + 1 + subplot_offset); % Create subplot and get handle
    copyobj(allchild(get(hfig_cell{j},'CurrentAxes')),subplot_cell{j}); axis off
    title(['t = ' num2str(x_data_windowed(j))]);
end
if print_img filename=regexprep(['Onset of ' conditions{condition_idx} ' Trials-MVGC-timeMontage'], ' ', '_'); print('-djpeg','-r600',[conn_folder filesep 'Images' filesep filename '.jpg']); end

%% Plot Spectral GC Graph:
figure('units','normalized','outerposition',[0 0 1 1]);
plot_spw_mod(full_mvgc_spec,fs,[],data_curated_labels);
% plot_spw_mod(full_mvgc_spec,fs,[],data_curated_labels,temp.full_mvgc_spec);
if print_img filename=regexprep(['Onset of ' conditions{condition_idx} ' Trials-MVGC-specMontage'], ' ', '_'); print('-djpeg','-r600',[conn_folder filesep 'Images' filesep filename '.jpg']); end

%% Plot Graph Theoretic parameters:
% % windowed_out_deg = cell2mat(cellfun(@(G)centrality(G,'outdegree','Importance',G.Edges.Weight),windowed_G,'un',0));
% windowed_out_deg = cell2mat(cellfun(@(G)centrality(G,'outdegree','Importance',G.Edges.Weight),windowed_G,'un',0));
% windowed_in_deg = cell2mat(cellfun(@(G)centrality(G,'indegree','Importance',G.Edges.Weight),windowed_G,'un',0));
% windowed_outin_deg = windowed_out_deg - windowed_in_deg;
% 
% figure; num_subplots = size(windowed_outin_deg,1);
% for i = 1:num_subplots
%     subplot(num_subplots,1,i);
%     plot(x_data_windowed, windowed_outin_deg(i,:) ,'color',node_colors(i,:)); hold on
%     % bar(x_data_windowed, windowed_outin_deg(i,:) ,'color',node_colors(i,:)); hold off
%     line([0 0],ylim,'Color','k'); line(xlim,[0 0],'Color','k','LineStyle','--');  
% end

% windowed_out_deg = cell2mat(cellfun(@(G)centrality(G,'outdegree','Importance',G.Edges.Weight),windowed_G,'un',0));
measure = 'degree';
max_in_column = 10;
sync_axes = 0;
compute_importanceDeg = 1;

if strcmp(measure,'degree')
    if compute_importanceDeg
        windowed_out_deg = cell2mat(cellfun(@(G)centrality(G,'outdegree','Importance',G.Edges.Weight),windowed_G,'un',0));
        windowed_in_deg = cell2mat(cellfun(@(G)centrality(G,'indegree','Importance',G.Edges.Weight),windowed_G,'un',0));
        file_abbv = 'OutInDegImportance';
    else
        windowed_out_deg = cell2mat(cellfun(@(G)centrality(G,'outdegree'),windowed_G,'un',0));
        windowed_in_deg = cell2mat(cellfun(@(G)centrality(G,'indegree'),windowed_G,'un',0));
        file_abbv = 'OutInDeg';
    end
    windowed_measure = windowed_out_deg - windowed_in_deg;
    
elseif strcmp(measure,'hubness')
    windowed_measure = cell2mat(cellfun(@(G)compute_hubness(G,compute_importanceDeg)',windowed_G,'un',0));
    file_abbv = [upper(measure(1)) measure(2:end)];   
    
else
    windowed_measure = cell2mat(cellfun(@(G)centrality(G,measure),windowed_G,'un',0));
    file_abbv = [upper(measure(1)) measure(2:end)];
end

if sync_axes
    global_ylim = [min(windowed_measure(:)) max(windowed_measure(:))];
end

figure; num_subplots = size(windowed_measure,1); num_columns = ceil(num_subplots/max_in_column);
for i = 1:num_subplots
%     if curr_column > max_in_column
%         column_idx = column_idx + 1;
%         curr_column = 0;
%     end
    subplot(max_in_column,num_columns,i);
    plot(x_data_windowed, windowed_measure(i,:) ,'color',node_colors(i,:),'LineWidth',1.5); hold on
    % bar(x_data_windowed, windowed_measure(i,:) ,'FaceColor',node_colors(i,:));
    
    if sync_axes curr_ylim = global_ylim;
    else curr_ylim = [min(windowed_measure(i,:)) max(windowed_measure(i,:))]; end
    line([0 0],curr_ylim,'Color','k'); line(xlim,[0 0],'Color','k','LineStyle','--');  
    % title(data_curated_labels{i});
    title(ROIs_toUse_shortlabels(data_curated_idx(i)));
    % curr_column = curr_column + 1;
end

if print_img filename=regexprep(['ROI Onset of ' conditions{condition_idx} ' Trials-MVGC-' file_abbv], ' ', '_'); print('-djpeg','-r600',[conn_folder filesep 'Images' filesep filename '.jpg']); end

%% Split into task blocks preceded by the same task or different task:
% Aggregate task blocks:
close all
tasks_to_aggregate = [1 2]; condition_idx = 1;

trial_onset_aggregate = cellfun(@(x) cat(1,x,repmat(tasks_to_aggregate(1),size(x))),curr_task_onset(:,1),'un',0); % Adding the task label as second row to task_onset vectors
for i = 2:length(tasks_to_aggregate)
    temp_cell = cellfun(@(x) cat(1,x,repmat(tasks_to_aggregate(i),size(x))),curr_task_onset(:,tasks_to_aggregate(i)),'un',0); % Adding the task label as second row to task_onset vectors
    trial_onset_aggregate = cellfun(@(x,y) cat(2,x,y), trial_onset_aggregate,temp_cell,'un',0);   
end
[~,trial_onset_aggregate_sortidx] = cellfun(@(x) sort(x,2), trial_onset_aggregate,'un',0);   
trial_labels_aggregate = cellfun(@(x,y) x(2,y(1,:)),trial_onset_aggregate,trial_onset_aggregate_sortidx,'un',0);
trial_labels_aggregate_diff = cellfun(@(x) diff(x),trial_labels_aggregate,'un',0);

% Find the trials that belong to the same condition_idx that have a different preceding condition
trial_labels_aggregate_diffidx = cellfun(@(x,y) find(y~=0)+1 ,trial_labels_aggregate,trial_labels_aggregate_diff,'un',0); % Need to change this for more than two "tasks_to_aggregate"
trial_labels_aggregate_sameidx = cellfun(@(x,y) find(y==0)+1 ,trial_labels_aggregate,trial_labels_aggregate_diff,'un',0); % Need to change this for more than two "tasks_to_aggregate"
% test = cellfun(@(x,y) x(y) ,trial_onset_aggregate_sorted,trial_labels_aggregate_diffidx,'un',0); 
trial_labels_aggregate_diff_taskidx = cellfun(@(x,y) (x(y) == condition_idx) ,trial_labels_aggregate,trial_labels_aggregate_diffidx,'un',0); 
trial_labels_aggregate_same_taskidx = cellfun(@(x,y) (x(y) == condition_idx) ,trial_labels_aggregate,trial_labels_aggregate_sameidx,'un',0); 

trial_onset_aggregate_sorted = cellfun(@(x,y) x(1,y(1,:)),trial_onset_aggregate,trial_onset_aggregate_sortidx,'un',0); 
trial_onset_aggregate_diff = cellfun(@(x,y,z) x(y(z)),trial_onset_aggregate_sorted,trial_labels_aggregate_diffidx,trial_labels_aggregate_diff_taskidx,'un',0);
trial_onset_aggregate_same = cellfun(@(x,y,z) x(y(z)),trial_onset_aggregate_sorted,trial_labels_aggregate_sameidx,trial_labels_aggregate_same_taskidx,'un',0);

trial_onsets_comb = trial_onset_aggregate_same;
start_idx_comb = cellfun(@(x) x - task_epoch_delta, trial_onsets_comb,'un',0);
end_idx_comb = cellfun(@(x) x + task_epoch_delta, trial_onsets_comb,'un',0); 

trial_comb = []; subject_trial_idx = [];
for i = 1:length(trial_onsets_comb) 
    idx_remove = start_idx_comb{i} <= 0; start_idx_comb{i}(idx_remove) = []; end_idx_comb{i}(idx_remove) = []; trial_onsets_comb{i}(idx_remove) = []; % Remove trials that start before the beginning
    idx_remove = end_idx_comb{i} > total_num_scans(i);start_idx_comb{i}(idx_remove) = []; end_idx_comb{i}(idx_remove) = []; trial_onsets_comb{i}(idx_remove) = []; % Remove trials that end after the last data point
    
    for j = 1:length(start_idx_comb{i})
        % trial_comb = cat(3,trial_comb, data_comb{i}(start_idx_comb{i}(j):end_idx_comb{i}(j),:));
        trial_comb = [trial_comb; {data_comb{i}(start_idx_comb{i}(j):end_idx_comb{i}(j),:)}];
        subject_trial_idx = [subject_trial_idx; i];
    end        
end

% Check for any trials that have all zeros and remove them (will give
% problems during further computation):
trial_comb_iszero = cell2mat(cellfun(@(x) sum(sum(x == 0,2)),trial_comb,'un',0)) > 0;
trial_comb = trial_comb(~trial_comb_iszero); subject_trial_idx = subject_trial_idx(~trial_comb_iszero);

%% Old code:

% Add the shaded error region:
% shadedErrorBar(1:size(trial_avg_comb,1),smooth(trial_avg_comb(:,DMN_ICA_comp)),smooth(trial_std_comb(:,DMN_ICA_comp)),'g');

% %% Average across Subjects:
% x = 0; nsub = [1:3 5:size(data,2)]; 
% for isub = 1:numel(nsub)
%     xt = [];
%     for icond = 1:size(conditions,2) 
%         if ~isempty(data{nsub(isub)}{icond}) 
%             xt = [xt data{nsub(isub)}{icond}(:,ICA_component)];
%         end
%     end
%     x = x + xt/size(data,2); % average across subjects
% end
% 
% figure; plot(x(:,1))

% %% Isolate the time course for the conditions/tasks:
% task_tc = cell(size(weights,1),size(weights,2),size(weights{1,1},2)); 
% task_tc_definition = cell(size(weights,1),size(weights,2),size(weights{1,1},2));
% task_tc_orig = cell(size(weights,1),size(weights,2),size(weights{1,1},2));
% in_orig = cell(size(weights,1),size(weights,2),size(weights{1,1},2));
% for isub = 1:size(weights,1)
%     % nsess = CONN_x.Setup.nsessions(min(length(CONN_x.Setup.nsessions),nsub));
%     rt = CONN_x.Setup.RT(min(numel(CONN_x.Setup.RT),isub))/10;
%     for icond = 1:size(weights,2)
%         parfor isess = 1:size(weights{1,1},2)
%             
%             % Calculated from the "weights" in ICA.timeseries.mat file
%             task_tc{isub, icond, isess} = weights{isub,icond}{isess} ~= 0; 
%             
%             % Calculated from the values defined in Setup.conditions:
%             task_tc_definition{isub, icond, isess} = CONN_x.Setup.conditions.values{isub}{icond}{isess};
%             
%             offs = ceil(100/rt);
%             onset = CONN_x.Setup.conditions.values{isub}{icond}{isess}{1}; val = ones(size(onset));
%             durat = CONN_x.Setup.conditions.values{isub}{icond}{isess}{2};
%             if numel(CONN_x.Setup.nscans)>=isub && numel(CONN_x.Setup.nscans{isub})>=isess
%                 x = zeros(offs + ceil(CONN_x.Setup.nscans{isub}{isess}*CONN_x.Setup.RT(min(numel(CONN_x.Setup.RT),isub))/rt),1);
%                 if length(durat) >= 1 
%                     in_accum = cell(1,length(onset));
%                     for n1 = 1:length(onset) 
%                         tdurat = max(rt,min(offs*rt+CONN_x.Setup.RT(min(numel(CONN_x.Setup.RT),isub))*CONN_x.Setup.nscans{isub}{isess}-onset(n1),durat(min(length(durat),n1))));
%                         in = offs + round(1+onset(n1)/rt+(0:tdurat/rt-1));
%                         x(in(in>0)) = val(n1);
%                         in_accum{n1} = [in_accum{n1}; in(in>0)/10];
%                     end
%                     in_orig{isub, icond, isess} = in_accum;
%                 end
%                 x = mean(reshape(x(offs+(1:10*CONN_x.Setup.nscans{isub}{isess})),[10,CONN_x.Setup.nscans{isub}{isess}]),1)';
%                 task_tc_orig{isub,icond,isess} = x;
%             end
%         
%         end
%     end
% end
% 
% curr_task_tc = task_tc(nsub,:,nsess);
% curr_in_orig = in_orig(nsub,:,nsess);
% curr_task_tc_orig = task_tc_orig(nsub,:,nsess);
% ABM_tc = curr_task_tc{1};
% WM_tc = curr_task_tc{2};
% figure; plot(ABM_tc*0.01); hold on; plot(WM_tc*0.01);

% %% Average across trials:
% trial_size_delta = 15; task_for_trial = WM_tc; % Can be ABM_tc or WM_tc
% trial_tc = zeros(1,trial_size_delta*2+1); trial_tc(trial_size_delta:end) = 1;
% trial_onsets = find(diff(task_for_trial) > 0);
% 
% start_idx = trial_onsets - trial_size_delta; end_idx = trial_onsets + trial_size_delta;
% start_idx(start_idx <= 0) = []; end_idx(start_idx <= 0) = []; trial_onsets(start_idx <= 0) = []; % Remove trials that start before the beginning
% start_idx(end_idx > size(data{nsub}{1},1)) = []; end_idx(end_idx > size(data{nsub}{1},1)) = []; trial_onsets(end_idx > size(data{nsub}{1},1)) = []; % Remove trials that end after the last data point
% trial = cell(1,numel(trial_onsets)); % Num_of_trials
% for i = 1:numel(trial_onsets)    
%     trial{i} = data{nsub}{1}(start_idx(i):end_idx(i),:) + data{nsub}{2}(start_idx(i):end_idx(i),:);    
% end
% trial_mat = cell2mat(permute(cellfun(@(x) reshape(x,1,size(trial{1},1),size(trial{1},2)),trial,'un',0),[2 1 3]));
% trial_avg = squeeze(mean(trial_mat,1));
% 
% figure; plot(~trial_tc*0.002); hold on; plot(trial_tc*0.002); 
% plot(smooth(trial_avg(:,DMN_ICA_comp))); plot(smooth(trial_avg(:,CEN_ICA_comp))); plot(smooth(trial_avg(:,SN_ICA_comp)));



% shadedErrorBar((1:size(trial_avg_comb,1))*TR',smooth(sum(trial_avg_comb(:,DMN_ICA_comp),2)), smooth(mean(trial_std_comb(:,DMN_ICA_comp),2)),[],1);
% 
% % figure; plot(~trial_tc*0.002); hold on; 
% %figure; plot(trial_tc*0.002); hold on; plot(nan(1,25)); % Not Normalized
% h = figure; plot(trial_tc*0.02); hold on; plot(nan(1,25)); % Normalized
% plot(smooth(sum(trial_avg_comb(:,DMN_ICA_comp),2))); plot(smooth(sum(trial_avg_comb(:,CEN_ICA_comp),2))); plot(smooth(sum(trial_avg_comb(:,SN_ICA_comp),2)));
% legend('ABM Task','','DMN Timecourse', 'CEN Timecouse', 'SN Timecourse');
% title('Onset of ABM Trials');
% plot_color = get(gca,'ColorOrder');


% %% Plot all of the sub-networks individually:
% % load('color_order_shades.mat'); % color_offset = 2;
% trial_avg_comb = mean(cat(3,trial_comb{:}),3); trial_ste_comb = std(cat(3,trial_comb{:}),[],3)./sqrt(length(trial_comb));
% unique_subnets = unique([net_idx{:}]); net_idx_done = zeros(length(net_idx),1);
% for i = 1:length(unique_subnets)
%     % Plot the figure:
%     x_data = (-task_epoch_delta:task_epoch_delta)*TR;
%     title_text = ['Onset of ' conditions{condition_idx} ' Trials-' ICA_labels{unique_subnets(i)}]; title_text = regexprep(title_text, '_', '');
%     % h = plot_network_timecourse(x_data,{ICA_labels{unique_subnets(i)}},title_text,{[1]},trial_avg_comb(:,unique_subnets(i)),trial_ste_comb(:,unique_subnets(i)),plot_colorord(i,:),plot_errdisp);
%     curr_net_idx = find(cellfun(@(x) ismember(unique_subnets(i),x),net_idx)); net_idx_done(curr_net_idx) = net_idx_done(curr_net_idx) + 1;
%     subplot(length(unique_subnets),1,i);
%     h = plot_network_timecourse_subplot(x_data,{ICA_labels{unique_subnets(i)}},title_text,{[1]},trial_avg_comb(:,unique_subnets(i)),trial_ste_comb(:,unique_subnets(i)),color_order{curr_net_idx*1,net_idx_done(curr_net_idx)*2},plot_errdisp);
%     if print_img filename=regexprep(title_text, ' ', '_'); print('-djpeg','-r500',[conn_folder filesep 'Images' filesep filename '.jpg']); end    
% end