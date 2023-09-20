% To plot group figure:

base_path_main = '/project/6006749/shaws5/Research_code/EEGnet/Main';
output_base_path_data = '/project/rrg-beckers/shaws5/Research_data'; %GRAHAM_OUTPUT_PATH
final_featuresToUse = 'individual'; % Can be 'preselected' or 'individual'
class_types = 'networks'; % Originally CONN_cfg.class_types, can be 'subnetworks' or 'networks' which will be based on the CONN_cfg.net_to_analyze

%% Get Fully supervised Results:
% Load fully supervised CohortA:
study_name = 'CompositeTask';
A_fullSup = load([base_path_main filesep 'plotFinalClassificationResultsAcc_' study_name '_FEAT' final_featuresToUse '_CLASS' class_types '_NEW'],'All_TestAccuracy_mean','All_TestAccuracy_null_mean');
A_fullSup.All_TestAccuracy_mean(A_fullSup.All_TestAccuracy_mean == 0) = nan;
A_fullSup_mean = round(nanmean(A_fullSup.All_TestAccuracy_mean,2)*100,1);
A_fullSup_null_mean = round(nanmean(A_fullSup.All_TestAccuracy_null_mean,2)*100,1);

% Load fully supervised CohortB:
study_name = 'AmyTasks';
B_fullSup = load([base_path_main filesep 'plotFinalClassificationResultsAcc_' study_name '_FEAT' final_featuresToUse '_CLASS' class_types '_NEW'],'All_TestAccuracy_mean','All_TestAccuracy_null_mean');
B_fullSup.All_TestAccuracy_mean(B_fullSup.All_TestAccuracy_mean == 0) = nan;
B_fullSup_mean = round(nanmean(B_fullSup.All_TestAccuracy_mean,2)*100,1);
B_fullSup_null_mean = round(nanmean(B_fullSup.All_TestAccuracy_null_mean,2)*100,1);

%% Get Generalized results:
features_to_include = [1 2 3 4 5];

% Load fully supervised CohortA:
study_name = 'CompositeTask'; sub_IDX = [2:15];
Results_outputDir = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'Classification_Results'];
A_gen_Testdata = nan(max(sub_IDX),1); A_gen_TestNulldata = nan(max(sub_IDX),1);
for i = sub_IDX
    A_gen = load([Results_outputDir filesep 'FinalResults_' study_name 'LOO' '_FEAT' 'preselected' '_CLASS' class_types 'NEW' '_Feat' arrayfun(@(x) num2str(x),features_to_include) '_CVrun' num2str(i)],'TestAccuracy_SUB','TestAccuracy_null_SUB');
    A_gen_Testdata(i) = A_gen.TestAccuracy_SUB(i);
    A_gen_TestNulldata(i) = A_gen.TestAccuracy_null_SUB(i);
end
A_gen_mean = round(nanmean(A_gen_Testdata,2)*100,1);
A_gen_null_mean = round(nanmean(A_gen_TestNulldata,2)*100,1);

% Load fully supervised CohortB:
study_name = 'AmyTasks'; sub_IDX = [2:13];
Results_outputDir = [output_base_path_data filesep 'Analyzed_data' filesep 'GroupResults_' study_name filesep 'Classification_Results'];
B_gen_Testdata = nan(max(sub_IDX),1); B_gen_TestNulldata = nan(max(sub_IDX),1);
for i = sub_IDX
    B_gen = load([Results_outputDir filesep 'FinalResults_' study_name 'LOO' '_FEAT' final_featuresToUse '_CLASS' class_types 'NEW' '_Feat' arrayfun(@(x) num2str(x),features_to_include) '_CVrun' num2str(i)],'TestAccuracy_SUB','TestAccuracy_null_SUB');
    B_gen_Testdata(i) = B_gen.TestAccuracy_SUB(i);
    B_gen_TestNulldata(i) = B_gen.TestAccuracy_null_SUB(i);
end
B_gen_mean = round(nanmean(B_gen_Testdata,2)*100,1);
B_gen_null_mean = round(nanmean(B_gen_TestNulldata,2)*100,1);

%% Plot the figure:
classificationType_text = {'Null','Generalized','Semi-Supervised','Fully-Supervised'};
printFig = 1;
fig_dataset_name = 'ALLSUB_CompositeTask';
printFig_location = [output_base_path_data filesep 'EEGnet_OutputImages'];

All_TestAccuracy_mean = [A_fullSup_null_mean(2:15) A_gen_mean(2:end) A_semSup_mean(2:end) A_fullSup_mean(2:15)];
figure('units','normalized','outerposition',[0 0 1 1]);
curr_plotData = All_TestAccuracy_mean; bar(curr_plotData); hold on; legend(classificationType_text,'Location','northeastoutside'); legend('boxoff'); ylim([0 100]); title('Test Accuracy');
if printFig print('-djpeg','-r500',[printFig_location filesep 'GroupTestAccuracy-A_' fig_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

A_curr_plotData_mean = nanmean(All_TestAccuracy_mean);
A_curr_plotData_std = nanstd(All_TestAccuracy_mean);

% figure('units','normalized','outerposition',[0 0 1 1]);
% A_curr_plotData = nanmean(All_TestAccuracy_mean); bar(curr_plotData); hold on; legend(legend_text,'Location','northeastoutside'); legend('boxoff'); ylim([0 100]); title('Test Accuracy');
% if printFig print('-djpeg','-r500',[printFig_location filesep 'GroupTestAccuracy_' fig_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

All_TestAccuracy_mean = [B_fullSup_null_mean(2:end) B_gen_mean(2:end) B_semSup_mean(2:end) B_fullSup_mean(2:end)];
figure('units','normalized','outerposition',[0 0 1 1]);
curr_plotData = All_TestAccuracy_mean; bar(curr_plotData); hold on; legend(classificationType_text,'Location','northeastoutside'); legend('boxoff'); ylim([0 100]); title('Test Accuracy');
if printFig print('-djpeg','-r500',[printFig_location filesep 'GroupTestAccuracy-B_' fig_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

B_curr_plotData_mean = nanmean(All_TestAccuracy_mean);
B_curr_plotData_std = nanstd(All_TestAccuracy_mean);

figure('units','normalized','outerposition',[0 0 1 1]); label_text = {'Cohort A', 'Cohort B'};
curr_plotData = [A_curr_plotData_mean' B_curr_plotData_mean']; bar(curr_plotData); hold on;errorbar_grouped(curr_plotData,[A_curr_plotData_std' B_curr_plotData_std']); legend(label_text,'Location','northeastoutside'); legend('boxoff'); ylim([0 100]); title('Test Accuracy'); xticklabels(classificationType_text);
if printFig print('-djpeg','-r500',[printFig_location filesep 'GroupTestAccuracyAll_' fig_dataset_name '_FEAT' final_featuresToUse '_CLASS' class_types]); close all; end

