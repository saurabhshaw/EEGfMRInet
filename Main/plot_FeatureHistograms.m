function plot_FeatureHistograms(printFig,plot_chanPercent,histData,mRMRiterateResults,plotParams,svmParams)

% Setup variables:
window_hist_all = histData.window_hist_all;
FreqA_hist_all = histData.FreqA_hist_all;
FreqB_hist_all = histData.FreqB_hist_all;
ChansA_hist_all = histData.ChansA_hist_all;
ChansB_hist_all = histData.ChansB_hist_all;

printFig_location = plotParams.printFig_location;
curr_dataset_name = plotParams.curr_dataset_name;
final_featuresToUse = plotParams.final_featuresToUse;
class_types = plotParams.class_types;
chanlocs = plotParams.chanlocs;
Feature_size = plotParams.Feature_size;
frequency_bands = plotParams.frequency_bands;
numFeats_per_featType = plotParams.numFeats_per_featType;

if isfield(svmParams,'TestAccuracy_CVruns') TestAccuracy_CVruns = svmParams.TestAccuracy_CVruns; end
if isfield(svmParams,'Model_CVruns') Model_CVruns = svmParams.Model_CVruns; end
Features = svmParams.Features;
YY_final = svmParams.YY_final;

%% Plot time window histograms:
window_bins = 5;
window_binSize = ceil(Feature_size(1)./window_bins);
window_bin_startIDX = 1:window_binSize:Feature_size(1); if length(window_bin_startIDX) > window_bins window_bin_startIDX(end) = []; end
window_bin_endIDX = window_bin_startIDX + window_binSize; if window_bin_endIDX(end) > Feature_size(1) window_bin_endIDX(end) = Feature_size(1); end
legendText = cellfun(@(x)regexprep(x,'_',' '),mRMRiterateResults.currFeatures_curated,'un',0);

window_hist_all_mat = window_hist_all;
window_hist_all_selected = zeros(window_bins,size(window_hist_all_mat,2)); window_hist_all_std_selected = zeros(window_bins,size(window_hist_all_mat,2)); xlabel_text = cell(1,window_bins);
for kk = 1:window_bins
    window_hist_all_selected(kk,:) = sum(window_hist_all_mat(window_bin_startIDX(kk):window_bin_endIDX(kk),:));
    if isfield(histData,'window_hist_all_std') window_hist_all_std_selected(kk,:) = mean(histData.window_hist_all_std(window_bin_startIDX(kk):window_bin_endIDX(kk),:)); end
    xlabel_text{kk} = [num2str(window_bin_startIDX(kk)) ' - ' num2str(window_bin_endIDX(kk))];
end
if plot_chanPercent window_hist_all_selected = (window_hist_all_selected./numFeats_per_featType)*100; window_hist_all_std_selected = (window_hist_all_std_selected./numFeats_per_featType)*100; end
figure; bar(window_hist_all_selected); xticklabels(xlabel_text); xlabel('Time windows'); legend(legendText,'Location','northeastoutside'); legend('boxoff');
if isfield(histData,'window_hist_all_std') hold on; errorbar_grouped(window_hist_all_selected,window_hist_all_std_selected); end
if printFig print('-djpeg','-r500',[printFig_location filesep 'WindowHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

%% Plot Frequency Bands:
FreqA_hist_all_mat = FreqA_hist_all; 
if isfield(histData,'FreqA_hist_all_std') FreqA_hist_all_std_mat = histData.FreqA_hist_all_std; end
if plot_chanPercent 
    FreqA_hist_all_mat = (FreqA_hist_all_mat./numFeats_per_featType)*100; 
    if isfield(histData,'FreqA_hist_all_std') FreqA_hist_all_std_mat = (FreqA_hist_all_std_mat./numFeats_per_featType)*100; end
end
figure; bar(FreqA_hist_all_mat); xticklabels(frequency_bands); xlabel('Frequency Bands'); legend(legendText,'Location','northeastoutside'); legend('boxoff');
if isfield(histData,'FreqA_hist_all_std') hold on; errorbar_grouped(FreqA_hist_all_mat,FreqA_hist_all_std_mat); end
if printFig print('-djpeg','-r500',[printFig_location filesep 'FreqAHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

FreqB_hist_all_mat = FreqB_hist_all;
if isfield(histData,'FreqB_hist_all_std') FreqB_hist_all_std_mat = histData.FreqB_hist_all_std; end
if plot_chanPercent 
    FreqB_hist_all_mat = (FreqB_hist_all_mat./numFeats_per_featType)*100; 
    if isfield(histData,'FreqB_hist_all_std') FreqB_hist_all_std_mat = (FreqB_hist_all_std_mat./numFeats_per_featType)*100; end
end
figure; bar(FreqB_hist_all_mat); xticklabels(frequency_bands); xlabel('Frequency Bands'); legend(legendText,'Location','northeastoutside'); legend('boxoff');
if isfield(histData,'FreqB_hist_all_std') hold on; errorbar_grouped(FreqB_hist_all_mat,FreqB_hist_all_std_mat); end
if printFig print('-djpeg','-r500',[printFig_location filesep 'FreqBHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

%% Plot Channel Topoplots:
% figure; topoplot2((chans_hist - mean(chans_hist))./std(chans_hist),1:51);
num_columns = 2; subplot_offset = 0;
numFeats = length(mRMRiterateResults.currFeatures_curated); subplotgrid = [numFeats num_columns];
for j = 1:numFeats
    % Plot the images:
    ChansA_hist = ChansA_hist_all{j};
    if plot_chanPercent ChansA_hist = (ChansA_hist./numFeats_per_featType)*100; end
    hfigCell_ChansA{j} = figure; topoplot(ChansA_hist, chanlocs); colormap(redblue); colorbar; lim = caxis; caxis([0 lim(2)]);
    
    ChansB_hist = ChansB_hist_all{j};
    if plot_chanPercent ChansB_hist = (ChansB_hist./numFeats_per_featType)*100; end
    hfigCell_ChansB{j} = figure; topoplot(ChansB_hist, chanlocs); colormap(redblue); colorbar; lim = caxis; caxis([0 lim(2)]);
    
end
h_composite = figure('units','normalized','outerposition',[0 0 0.4 1]);
for j = 1:numFeats
    % Accumulate the images into a composite image:
    subplotCellA{j} = subplot(subplotgrid(1), subplotgrid(2), 1 + ((j-1)*num_columns) + subplot_offset); % Create subplot and get handle
    copyobj(allchild(get(hfigCell_ChansA{j},'CurrentAxes')),subplotCellA{j}); colormap(redblue); colorbar; axis off
    if j == 1 title('ChansA'); end % REPLACE WITH SOURCE OR SINK NOMENCLATURE
    subplotCellA{j}.YAxis.Label.Color = [0 0 0]; subplotCellA{j}.YAxis.Label.Visible = 'on';
    ylabel(regexprep(mRMRiterateResults.currFeatures_curated{j}, '_', ' '),'Interpreter', 'none');
    
    subplotCellB{j} = subplot(subplotgrid(1), subplotgrid(2), 1 + ((j-1)*num_columns) + subplot_offset + 1); % Create subplot and get handle
    copyobj(allchild(get(hfigCell_ChansB{j},'CurrentAxes')),subplotCellB{j}); colormap(redblue); colorbar; axis off
    if j == 1 title('ChansB'); end % REPLACE WITH SOURCE OR SINK NOMENCLATURE
end
if printFig print('-djpeg','-r500',[printFig_location filesep 'ChansHist_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

%% Visualize Support Vectors:
% Model = classificationResults.Model_SUB{ii,tt};

% switch model_type
%     case 'SVM_libsvm'
% [~,topCV_run] = max(classificationResults.TestAccuracy(tt,:));
% TestAccuracy_CVruns = classificationResults.TestAccuracy(tt,:);
if svmParams.visualizeSVs
    if isfield(svmParams,'TestAccuracy_CVruns') [~,topCV_run] = max(TestAccuracy_CVruns); end
    % Model = classificationResults.Model;
    if isfield(svmParams,'Model_CVruns') sv = Model_CVruns{topCV_run}.SVs; sv = []; else; sv = []; end
    visualize_SVM_SVs_tSNE(Features, YY_final, sv, 3)
    if printFig print('-djpeg','-r500',[printFig_location filesep 'VisualizeSV_' curr_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end
end
% end