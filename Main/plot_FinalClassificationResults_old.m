% Use this script to plot the final classification results of a "process_*studyName*" file

study_name = 'CompositeTask';
modality = 'EEGfMRI';
chanlocs_fileName = 'BrainProductsMR64_NZ_LPA_RPA_fixed.sfp';

final_featuresToUse = 'individual'; % Can be 'preselected' or 'individual'
class_types = 'subnetworks'; % Originally CONN_cfg.class_types, can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze
runs_to_include = {'task'};
remove_ECGchannel = 1;

% For getting labels (SPECIFIC TO THIS DATASET):
% Setup CONN_cfg:
CONN_cfg = [];
CONN_cfg.CONN_analysis_name = 'ROI'; % The name of the CONN first-level analysis
% CONN_cfg.CONN_analysis_name = 'V2V_02'; % The name of the CONN first-level analysis
% CONN_cfg.CONN_project_name = 'conn_composite_task_fMRI'; % The name of the CONN project
CONN_cfg.CONN_project_name = 'conn_composite_task_fMRI_corrected'; % The name of the CONN project
CONN_cfg.CONN_data_type = 'ICA'; % The source of the CONN data - can be ICA or ROI 
CONN_cfg.net_to_analyze = {'CEN', 'DMN', 'SN'}; % List all networks to analyze
CONN_cfg.use_All_cond = 1; % If this is 1, use the timecourses from condition 'All'
CONN_cfg.p_norm = 0; % Lp norm to use to normalize the timeseries data:  For data normalization - set to 0 to skip it, 1 = L1 norm and 2 = L2 norm
CONN_cfg.conditions_to_include = [1 2]; % The condition indices to sum up in the norm
CONN_cfg.window_step = 2; % in seconds - Used for Feature windows:
CONN_cfg.window_length = 10; % in seconds - Used for Feature windows:
CONN_cfg.threshold = 0.5; % Threshold for making the time course binary
CONN_cfg.class_types = 'subnetworks'; % Can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze
CONN_cfg.multilabel = 0; % Whether classification is multilabel or single label
CONN_cfg.ROIs_toUse = {'ICA_CEN','ICA_LCEN','ICA_anteriorSN','ICA_posteriorSN','ICA_ventralDMN','ICA_dorsalDMN'}; % Need this if using ROIs for labels rather than ICA
CONN_cfg.rescale = 1; % Rescale between 0 and 1 if selected

% Study specific parameters:
study_conditions = {'ABM','WM','Rest','ABM-WM','WM-ABM'};
replace_files = 0;
scan_parameters = [];
scan_parameters.TR = 2; % MRI Repetition Time (in seconds)
scan_parameters.anat_num_images = 184;
scan_parameters.rsfunc_num_images = 5850;
scan_parameters.tfunc_num_images = 11700;
scan_parameters.slicespervolume = 39;
scan_parameters.slice_marker = 'R128';
scan_parameters.ECG_channel = 32;
scan_parameters.srate = 5000;
scan_parameters.low_srate = 500;

% Set Paths:
% base_path_main = fileparts(mfilename('fullpath')); cd(base_path_main); cd ..;
% base_path = pwd;
base_path_main = fileparts(matlab.desktop.editor.getActiveFilename); cd(base_path_main); cd ..
base_path = pwd;

% Include path:
toolboxes_path = [base_path filesep 'Toolboxes']; base_path_main = [base_path filesep 'Main'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB'];
addpath(genpath(base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));
addpath(genpath([toolboxes_path filesep 'libsvm-3.23'])); % Adding LibSVM
% addpath(genpath([toolboxes_path filesep 'libsvm-3.23' filesep 'windows'])); % Adding LibSVM
addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib
addpath(genpath([toolboxes_path filesep 'Mricron'])) % Adding Mricron
addpath(genpath([toolboxes_path filesep 'dimredtoolbox'])) % Adding Dimredtoolbox for TSNE

% Setup Data paths:
[base_path_rc, base_path_rd] = setPaths();
base_path_data = base_path_rd;
output_base_path_data = '/project/rrg-beckers/shaws5/Research_data'; %GRAHAM_OUTPUT_PATH
% output_base_path_data = base_path_data; %GRAHAM_OUTPUT_PATH
offline_preprocess_cfg.temp_file = [output_base_path_data filesep 'temp_deploy_files'];

%% Get number of runs/sessions/etc:
[sub_dir,sub_dir_mod] = update_subject_list(study_name,modality,base_path_data,runs_to_include);

% SPECIFIC TO THIS DATASET:
CONN_cfg.CONNresults_folder = [base_path filesep 'Models' filesep 'ICA_Labels' filesep CONN_cfg.CONN_project_name '-' CONN_cfg.CONN_analysis_name];
CONN_data = conn_loadROI_data(CONN_cfg,scan_parameters);

EEGfMRI_corr = readtable([base_path_rd filesep 'EEGfMRI_ProcessingCorrespondence_' study_name '.xls']); % This is to make sure that the blocks of the EEG and fMRI correspond together
EEGfMRI_corr_firstblockIDX = find(cellfun(@(x) strcmp(x,'EEGTaskBlocks'),EEGfMRI_corr.Properties.VariableNames));
EEGfMRI_corr_validIDX = cellfun(@(x)~isnan(x),table2cell(EEGfMRI_corr(:,1))); EEGfMRI_corr = EEGfMRI_corr(EEGfMRI_corr_validIDX,:);
EEGfMRI_corrIDX = zeros(sum(EEGfMRI_corr_validIDX),(size(EEGfMRI_corr,2)-EEGfMRI_corr_firstblockIDX+1));
for i = 1:size(EEGfMRI_corr,1)
    temp_corrIDX = table2cell(EEGfMRI_corr(i,EEGfMRI_corr_firstblockIDX:size(EEGfMRI_corr,2)));
    for j = 1:length(temp_corrIDX)
        if ischar(temp_corrIDX{j}) temp_corrIDX{j} = str2num(temp_corrIDX{j}); end
        if isempty(temp_corrIDX{j}) temp_corrIDX{j} = NaN; end
    end
    EEGfMRI_corrIDX(i,:) = cell2mat(temp_corrIDX);
end

%% Get Feature processing settings:
% Read settings from file
fileID = fopen([base_path filesep 'Features' filesep 'Compute_features_scripts' filesep 'compute_features_deploy_settings.txt'],'r');
data = textscan(fileID,'%s','Delimiter','\n');
fclose(fileID);

% Execute statements:
frequencybands_IDX = find(cellfun(@(x)strcmp(x,'% Options for frequency bands to compute features over:'),data{1})) + 2;
eval(data{1}{frequencybands_IDX});
% for line_num = 1:length(data{1})
%     eval(data{1}{line_num});    
% end

% CUSTOM FOR THIS RUN ONLY:
if length(frequency_bands) > 6 frequency_bands(end) = []; end

%% Final data processing:
load([base_path_main filesep 'FinalResults_' study_name '_FEAT' final_featuresToUse '_CLASS' class_types]); 
plotResults_individual = 1; jj = 1;
printFig = 0; printFig_location = [output_base_path_data filesep 'EEGnet_OutputImages'];

% Load channel locations:
chanlocs_file = [base_path filesep 'Cap_files' filesep chanlocs_fileName];
chanlocs_raw = pop_chanedit([],'load',{chanlocs_file 'filetype' 'autodetect'});
if remove_ECGchannel
    ECG_channel = find(cellfun(@(x) strcmp(x,'ECG'),{chanlocs_raw.labels}));
    chanlocs = chanlocs_raw; chanlocs(ECG_channel) = [];
end

if (printFig && isempty(dir(printFig_location))) mkdir(printFig_location); end

% Get the final accuracy data:
All_TestAccuracy_mean = nan(size(EEGfMRI_corrIDX)); All_TestAccuracy_null_mean = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_mean = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_null_mean = nan(size(EEGfMRI_corrIDX));
All_TestAccuracy_std = nan(size(EEGfMRI_corrIDX)); All_TestAccuracy_null_std = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_std = nan(size(EEGfMRI_corrIDX)); All_TrainAccuracy_null_std = nan(size(EEGfMRI_corrIDX));

%% Run Loop
All_TrainAccuracy_ttest = []; All_TestAccuracy_ttest = [];
for ii = 2:length(All_TestAccuracy)
    temp_idx = ~isnan(EEGfMRI_corrIDX(ii,:));
    All_TestAccuracy_mean(ii,temp_idx) = mean(All_TestAccuracy{ii}(temp_idx,:),2);
    All_TestAccuracy_null_mean(ii,temp_idx) = mean(All_TestAccuracy_null{ii}(temp_idx,:),2);
    All_TrainAccuracy_mean(ii,temp_idx) = mean(All_TrainAccuracy{ii}(temp_idx,:),2);
    All_TrainAccuracy_null_mean(ii,temp_idx) = mean(All_TrainAccuracy_null{ii}(temp_idx,:),2);
    
    All_TestAccuracy_std(ii,temp_idx) = std(All_TestAccuracy{ii}(temp_idx,:),[],2);
    All_TestAccuracy_null_std(ii,temp_idx) = std(All_TestAccuracy_null{ii}(temp_idx,:),[],2);
    All_TrainAccuracy_std(ii,temp_idx) = std(All_TrainAccuracy{ii}(temp_idx,:),[],2);
    All_TrainAccuracy_null_std(ii,temp_idx) = std(All_TrainAccuracy_null{ii}(temp_idx,:),[],2);
    
    % Test if the Accuracies of the Null model and actual model are different: 
    for tt = find(temp_idx)
        [All_TrainAccuracy_ttest{ii,tt}.h,All_TrainAccuracy_ttest{ii,tt}.p,All_TrainAccuracy_ttest{ii,tt}.ci,All_TrainAccuracy_ttest{ii,tt}.stats] = ttest2(All_TrainAccuracy{ii}(tt,:),All_TrainAccuracy_null{ii}(tt,:));
        [All_TestAccuracy_ttest{ii,tt}.h,All_TestAccuracy_ttest{ii,tt}.p,All_TestAccuracy_ttest{ii,tt}.ci,All_TestAccuracy_ttest{ii,tt}.stats] = ttest2(All_TestAccuracy{ii}(tt,:),All_TestAccuracy_null{ii}(tt,:));
        
        
        % Plot histograms of the top features:
        if plotResults_individual
            curr_CONN_IDX = EEGfMRI_corrIDX(ii,tt);
            dataset_to_use = [sub_dir(ii).name];
            dataset_name = [sub_dir_mod(ii).PID];
            curr_dir = [output_base_path_data filesep dataset_to_use];
            task_dir = [curr_dir filesep 'Task_block_' num2str(tt)];
            Featurefiles_directory = [task_dir filesep 'EEG_Features']; curr_dataset_name = [runs_to_include{jj} '_' dataset_name '_VHDR_TaskBlock' num2str(tt)];
            Featurefiles_basename = ['Rev_' curr_dataset_name];
            
            mRMRiterateResults = load([Featurefiles_directory filesep Featurefiles_basename '_mRMRiterateResults.mat']);
            
            % Get naming convention and features:
            switch final_featuresToUse
                case 'individual'
                    name_suffix = '';
                    Features = mRMRiterateResults.final_dataset_mRMR;
                case 'preselected'
                    name_suffix = 'GEN';
                    temp = load([Featurefiles_directory filesep Featurefiles_basename '_FeaturesGEN.mat']);
                    Features = temp.Features_SUB;
            end
            
            classificationResults = load([Featurefiles_directory filesep Featurefiles_basename '_ClassificationResults' name_suffix '_FEAT' final_featuresToUse '_CLASS' class_types]);
            YY_final = cell2mat(CONN_data.fMRI_labels_selected_window_avg_thresh{ii-1}{curr_CONN_IDX});  % NOTE:only because this is excluding the first subject
            
            % Get the label breakdown:
            numFeats_per_featType = round(length(mRMRiterateResults.final_feature_labels_mRMR)./length(mRMRiterateResults.currFeatures_curated)); % This needs to be removed for the GEN case
            all_Feature_labels = cellfun(@(x)strsplit(x,'_'),mRMRiterateResults.final_feature_labels_mRMR,'UniformOutput',0);
            
            % features_vect = cell2mat(cellfun(@(x)str2num(x{1}),all_Feature_labels,'UniformOutput',0));
            window_vect = cell2mat(cellfun(@(x)str2num(x{1}),all_Feature_labels,'UniformOutput',0));
            freqBand_A_vect = cell2mat(cellfun(@(x)str2num(x{2}),all_Feature_labels,'UniformOutput',0));
            freqBand_B_vect = cell2mat(cellfun(@(x)str2num(x{end-2}),all_Feature_labels,'UniformOutput',0));
            chan_A_vect = cell2mat(cellfun(@(x)str2num(x{end-1}),all_Feature_labels,'UniformOutput',0));
            chan_B_vect = cell2mat(cellfun(@(x)str2num(x{end}),all_Feature_labels,'UniformOutput',0));
            
            % Plot these parameters for each feature:
            numFeats = length(mRMRiterateResults.currFeatures_curated);
            features_vect = []; % window_hist = zeros(Feature_size(1),numFeats);
            window_hist_all = cell(1,numFeats); FreqA_hist_all = cell(1,numFeats); FreqB_hist_all = cell(1,numFeats);
            ChansA_hist_all = cell(1,numFeats); ChansB_hist_all = cell(1,numFeats);
            % FreqA_hist_all = []; FreqB_hist_all = [];
            for k = 1:numFeats
                
                % Get size of the feature:
                load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' mRMRiterateResults.currFeatures_curated{k} '.mat'],'Feature_size');
                
                % Find current features:
                featIDX_start = (k-1)*numFeats_per_featType + 1; featIDX_end = featIDX_start + numFeats_per_featType - 1;
                Feature_labels_IDX = zeros(1,length(all_Feature_labels)); Feature_labels_IDX(featIDX_start:featIDX_end) = 1;
                features_vect = [features_vect repmat(k,[1 numFeats_per_featType])];
                curr_freqBand_B_vect = freqBand_B_vect; if length(Feature_size) < 5 curr_freqBand_B_vect(~Feature_labels_IDX) = NaN; end                
               
                % Window: Compute Histogram -  
                window_hist = zeros(Feature_size(1),1); 
                for i = 1:Feature_size(1) window_hist(i) = sum(window_vect(featIDX_start:featIDX_end) == i); end
                window_hist_all{k} = window_hist;
                % bar(window_hist);

                % FreqA: Compute Histogram -
                FreqA_hist = zeros(Feature_size(2),1);
                for i = 1:Feature_size(2) FreqA_hist(i) = sum(freqBand_A_vect(featIDX_start:featIDX_end) == i); end
                % FreqA_hist_all = cat(2,FreqA_hist_all,FreqA_hist);
                FreqA_hist_all{k} = FreqA_hist;
                % bar(FreqA_hist); 

                % FreqB: Compute Histogram -
                if length(Feature_size) == 5                    
                    FreqB_hist = zeros(Feature_size(end-2),1);
                    for i = 1:Feature_size(end-2) FreqB_hist(i) = sum(curr_freqBand_B_vect(featIDX_start:featIDX_end) == i); end
                    % FreqB_hist_all = cat(2,FreqB_hist_all,FreqB_hist);
                    FreqB_hist_all{k} = FreqB_hist;
                    % bar(FreqB_hist);
                else
                    FreqB_hist_all{k} = zeros(Feature_size(2),1);
                end
                
                % ChansA: Compute Histogram -  
                ChansA_hist = zeros(Feature_size(end-1),1);
                for i = 1:Feature_size(end-1) ChansA_hist(i) = sum(chan_A_vect(featIDX_start:featIDX_end) == i); end
                ChansA_hist_all{k} = ChansA_hist;
                figure; topoplot(ChansA_hist, chanlocs); colormap(redblue); colorbar; lim = caxis; caxis([0 lim(2)]);
                % bar(ChansA_hist); % figure; topoplot2((chans_hist - mean(chans_hist))./std(chans_hist),1:51);
                if printFig print('-djpeg','-r500',[printFig_location filesep 'ChansAHist_' mRMRiterateResults.currFeatures_curated{k} '_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

                % ChansB: Compute Histogram -  
                ChansB_hist = zeros(Feature_size(end),1);
                for i = 1:Feature_size(end) ChansB_hist(i) = sum(chan_B_vect(featIDX_start:featIDX_end) == i); end
                ChansB_hist_all{k} = ChansB_hist;
                figure; topoplot(ChansB_hist, chanlocs); colormap(redblue); colorbar; lim = caxis; caxis([0 lim(2)]);
                % bar(ChansB_hist); % figure; topoplot2((chans_hist - mean(chans_hist))./std(chans_hist),1:51);
                if printFig print('-djpeg','-r500',[printFig_location filesep 'ChansBHist_' mRMRiterateResults.currFeatures_curated{k} '_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

            end
            
            % Plot time window histograms:
            window_bins = 5;
            window_binSize = ceil(Feature_size(1)./window_bins);
            window_bin_startIDX = 1:window_binSize:Feature_size(1); if length(window_bin_startIDX) > window_bins window_bin_startIDX(end) = []; end
            window_bin_endIDX = window_bin_startIDX + window_binSize; if window_bin_endIDX(end) > Feature_size(1) window_bin_endIDX(end) = Feature_size(1); end
            
            window_hist_all_mat = cell2mat(window_hist_all);
            window_hist_all_selected = zeros(window_bins,size(window_hist_all_mat,2)); xlabel_text = cell(1,window_bins);
            for kk = 1:window_bins
                window_hist_all_selected(kk,:) = sum(window_hist_all_mat(window_bin_startIDX(kk):window_bin_endIDX(kk),:));
                xlabel_text{kk} = [num2str(window_bin_startIDX(kk)) ' - ' num2str(window_bin_endIDX(kk))];
            end            
            figure; bar(window_hist_all_selected); xticklabels(xlabel_text);
            if printFig print('-djpeg','-r500',[printFig_location filesep 'WindowHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

            % Plot Frequency Bands:
            figure; bar(cell2mat(FreqA_hist_all)); xticklabels(frequency_bands);
            if printFig print('-djpeg','-r500',[printFig_location filesep 'FreqAHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

            figure; bar(cell2mat(FreqB_hist_all)); xticklabels(frequency_bands);
            if printFig print('-djpeg','-r500',[printFig_location filesep 'FreqBHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

            % Plot Channel Topoplots:
            % figure; topoplot2((chans_hist - mean(chans_hist))./std(chans_hist),1:51);
            
            % Visualize Support Vectors:
            % Model = classificationResults.Model_SUB{ii,tt}; 
            
            % switch model_type
            %     case 'SVM_libsvm' 
            [~,topCV_run] = max(classificationResults.TestAccuracy(tt,:));
            Model = classificationResults.Model;
            sv = Model{topCV_run}.SVs; sv = [];
            visualize_SVM_SVs_tSNE(Features, YY_final, sv, 3)
            if printFig print('-djpeg','-r500',[printFig_location filesep 'VisualizeSV_' curr_dataset_name]); close all; end
            % end
        end
    end
end

%% Old Code:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%             features_vect = [];
%             for k = 1:length(mRMRiterateResults.currFeatures_curated)
%                 features_vect = [features_vect repmat(k,[1 numFeats_per_featType])];
% %                 % Find the features corresponding to the current feature:
% %                 featIDX_start = (k-1)*numFeats_per_featType + 1; featIDX_end = featIDX_start + numFeats_per_featType - 1;
% %                 Feature_labels_IDX = zeros(1,length(all_Feature_labels));
% %                 Feature_labels_IDX(featIDX_start:featIDX_end) = 1;
% %                 Feature_labels = all_Feature_labels(Feature_labels_IDX);
% %                 % Feature_labels_IDX = cellfun(@(x) str2num(x{1}),all_Feature_labels) == i; % This is only if the first value in the label is the feature index
% %                 % Feature_labels = cellfun(@(x)x(2:end),all_Feature_labels(Feature_labels_IDX),'un',0); % This is only if the first value in the label is the feature index
% %                 
% %                 curr_window = []; curr_freqBand_A = []; curr_freqBand_B = [];
% %                 curr_chan_A = []; curr_chan_B = [];
% %                 
% %                 % Get the subscript indices:
% %                 for j = 1:length(Feature_labels)
% %                     curr_window = [curr_window str2num(Feature_labels{j}{1})];
% %                     curr_freqBand_A = [curr_freqBand_A str2num(Feature_labels{j}{2})];
% %                     curr_chan_A = [curr_chan_A str2num(Feature_labels{j}{end-1})];
% %                     curr_chan_B = [curr_chan_B str2num(Feature_labels{j}{end})];
% %                     
% %                     if length(Feature_labels{j}) > 4
% %                         curr_freqBand_B = [curr_freqBand_B str2num(Feature_labels{i}{3})];
% %                     else
% %                         curr_freqBand_B = [curr_freqBand_B NaN];
% %                     end
% %                 end
%                 
%             end



% For multiple grouped 3D bar graph:
%at x=1
a=[11;7;14;11;43;38];
b=[11;13;17;13;51;46];
c=[9;11;20;9;69;76];
y1=[a b c];
%at x=2
d=[38;61;75;38;28;33];
e=[46;132;135;88;36;51];
f=[76;186;180;115;85;72];
y2=[d e f];
figure(1); 
hold on;
%First x value
xval = 1; 
h = bar3(y1,'grouped');
Xdat = get(h,'Xdata');
for ii=1:length(Xdat)
    Xdat{ii}=Xdat{ii}+(xval-1)*ones(size(Xdat{ii}));
    set(h(ii),'XData',Xdat{ii});
end
%Second x value
xval = 2;
h = bar3(y2,'grouped');
Xdat = get(h,'Xdata');
for ii=1:length(Xdat)
    Xdat{ii}=Xdat{ii}+(xval-1)*ones(size(Xdat{ii}));
    set(h(ii),'XData',Xdat{ii});
end
xlim([0 3]);
view(3);
title('Grouped Style')
xlabel('x');
ylabel('y');
zlabel('z');