% Run this script for manually rejecting components -
% Stage5/7 of offline_preprocess_manual
%

file_toLoad = '*Stage*-Workspace*.mat';
[curr_filename,curr_dir] = uigetfile({file_toLoad},['Select the Stage file']);
load([curr_dir curr_filename],'curr_dir_preprocessed','dataset_name');
curr_dir_preprocessed_current = curr_dir;
stageCompletion_file = [curr_dir_preprocessed_current filesep dataset_name '_StageCompletion.mat']; load(stageCompletion_file)

current_stage_temp = strsplit(curr_filename,{[dataset_name '_Stage'], '-Workspace'}); current_stage = str2num(current_stage_temp{end-1})+1;

if max_finishedStage == current_stage-1
    if exist('uiprogressdlg')
        f = uifigure;  set(f,'menubar','none'); set(f,'NumberTitle','off'); d = uiprogressdlg(f,'Title','Loading File ...','Indeterminate','on');
    else
        disp('Loading File ...')
    end
    load([curr_dir curr_filename]); if exist('uiprogressdlg') close(d); end
    % current_stage_temp = strsplit(curr_filename,{[dataset_name '_Stage'], '-Workspace'}); current_stage = str2num(current_stage_temp{end-1})+1;
    if current_stage == 5 manualrejcomp_idx = 1; elseif current_stage == 7 manualrejcomp_idx = 2; end
    
    % eeglab redraw
    pop_viewprops(EEG,0,[1:size(EEG.icaweights,1)],{'freqrange', [0 50]});
    
    dlg_cfg = []; dlg_cfg.WindowStyle = 'normal';
    answer = inputdlg('ICA component numbers you want to reject (separated by spaces):','Manual ICA component rejection', [1 50],{''},dlg_cfg);
    EEG_ICA_Manualrejcomp{manualrejcomp_idx} = str2num(answer{1}); % Type all the component numbers you want to reject
    EEG = pop_subcomp(EEG, EEG_ICA_Manualrejcomp{manualrejcomp_idx}, 0);
    
    EEG.setname = regexprep(EEG.setname,' pruned with ICA',['_ICAmanreject' num2str(manualrejcomp_idx)]);
    EEG = pop_saveset( EEG, 'filename',['Stage' num2str(current_stage) '-' EEG.setname '.set'],'filepath',curr_dir_preprocessed_current);
    
    % Update preprocessing_stageCompletion
    preprocessing_stageCompletion(current_stage) = 1; max_finishedStage = max(find(preprocessing_stageCompletion));
    [~,stageCompletion_filename,~] = fileparts(stageCompletion_file);
    save([curr_dir_preprocessed_current stageCompletion_filename '.mat'],'preprocessing_stageCompletion','max_finishedStage');
    
    % Save breakpoint after updating preprocessing_stageCompletion
    if exist('uiprogressdlg')
        f = uifigure; d = uiprogressdlg(f,'Title','Writing Output File ...','Indeterminate','on');
    else
        disp('Writing Output File ...')
    end
    save([curr_dir_preprocessed_current dataset_name '_Stage' num2str(current_stage) '-Workspace'],'-regexp', '^(?!(current_stage|preprocessing_stageCompletion|max_finishedStage)$).');
    if exist('uiprogressdlg') close(d); end
    close all
else
    f = errordlg('Previous stages not completed. Please run them and try again','Stage Error');
end
