% Run this to add newly acquired subjects to the CompositeTask_EEGfMRI
% analysis:
function [sub_dir, sub_dir_mod] = conn_addSubjects_EEGfMRI(study_name,modality,base_path_data, scan_parameters,study_conditions, replace_files)

% % Control parameters:
% study_base_name = 'composite_task_';
% studyFolder_base_name = 'CompositeTask_EEGfMRI';
% excel_name = ['Subject_Info_' studyFolder_base_name];
% 
% % Study specific parameters:
% study_conditions = {'ABM','WM','Rest','ABM-WM','WM-ABM'};
% TR = 2; % MRI Repetition Time (in seconds)
% replace_files = 0;
% scan_parameters = [];
% scan_parameters.anat_num_images = 184;
% scan_parameters.rsfunc_num_images = 5850;
% scan_parameters.tfunc_num_images = 11700;
TR = scan_parameters.TR;

temp_study_name = find(isstrprop(study_name,'upper')); study_base_name = '';
for i = 1:length(temp_study_name)
    str_start_idx = temp_study_name(i); 
    if (i+1) > length(temp_study_name) str_end_idx = length(study_name); else str_end_idx = temp_study_name(i+1) - 1; end
    if i == 1 study_base_name = lower(study_name(str_start_idx:str_end_idx)); else study_base_name = [study_base_name '_' lower(study_name(str_start_idx:str_end_idx))]; end
end
study_base_name = [study_base_name '_'];
studyFolder_base_name = [study_name '_' modality];
% excel_name = ['Subject_Info_' studyFolder_base_name];
excel_name = ['Subject_Info_' study_name];

% Find all subject data directories that match this study_base_name:
% Get directory information:
sub_dir = dir([base_path_data filesep '*_' study_name '_' modality '*']);
sub_dir_mod = rmfield(sub_dir,{'folder','date','bytes','isdir','datenum'});

% Get processing information:
subject_folder_vect = {sub_dir.name};
temp_PID = cellfun(@(x) strsplit(x,{'-','_'}),subject_folder_vect,'un',0); temp_PID = cellfun(@(x) x{4},temp_PID,'un',0); [sub_dir_mod.PID] = temp_PID{:};
temp_SID = cellfun(@(x) strsplit(x,{'_','.mat'}),subject_folder_vect,'un',0); temp_SID_bin = cellfun(@(x) length(x) < 4 ,temp_SID); 
temp_SID(temp_SID_bin) = cellfun(@(x) [x {''}],temp_SID(temp_SID_bin),'un',0); temp_SID = cellfun(@(x) x{4},temp_SID,'un',0); [sub_dir_mod.SID] = temp_SID{:};

% sub_dir = dir(base_path_data); curr_dir_name = {sub_dir.name};
% match_dir_name = contains(curr_dir_name,studyFolder_base_name);
% subject_folder_vect = curr_dir_name(match_dir_name);

%% Get Subject Info:
% Get subjectID column:
subjectID_vect = cellfun(@(x)strsplit(x,'-'),subject_folder_vect,'UniformOutput',0); 
subjectID_vect = cellfun(@(x)x{2},subjectID_vect,'UniformOutput',0);

% Get subjectNumber and sessionNumber column:
[~,s2u_idx,~] = unique(subjectID_vect);
tmp = false(size(s2u_idx)); tmp(s2u_idx) = true;
unique_subjects = subjectID_vect(tmp); 

unique_subjects_num = num2cell(1:length(unique_subjects));
subjectNumber_vect_idx = cellfun(@(x)contains(subjectID_vect,x),unique_subjects,'UniformOutput',0); 
subjectNumber_vect = cell2mat(cellfun(@(x,y) nonzeros((x)*y),subjectNumber_vect_idx,unique_subjects_num,'UniformOutput',0));
sessionNumber_vect = cell2mat(cellfun(@(x) 1:sum(x),subjectNumber_vect_idx,'UniformOutput',0));

% Get Date column:
raw_date_vect = cellfun(@(x)strsplit(x,'_'),subject_folder_vect,'UniformOutput',0);
raw_date_vect = cellfun(@(x)x{1},raw_date_vect,'UniformOutput',0);
date_vect = cellfun(@(x)[x(1:4) '-' x(5:6) '-' x(7:8)],raw_date_vect,'UniformOutput',0);

% Get Exam number:
exam_vect_dir = cellfun(@(x) dir([base_path_data filesep x filesep 'MRI']),subject_folder_vect,'UniformOutput',0);
exam_vect_dir_bin = cellfun(@(x) ~isempty(x) ,exam_vect_dir); 
exam_vect = cellfun(@(x) strsplit(x(find(~[x.isdir])).name,{'_','.'}),exam_vect_dir(exam_vect_dir_bin),'UniformOutput',0);
exam_vect = cell2mat(cellfun(@(x)str2num(x{2}),exam_vect,'UniformOutput',0));

%% Iterate through all the found folders:
subject_folder_vect = subject_folder_vect(exam_vect_dir_bin);
for i = 1:length(subject_folder_vect)
    subject_folder = subject_folder_vect{i};
    subject_dir = [base_path_data filesep subject_folder];
    curr_exam = exam_vect(i); curr_ID = subjectID_vect{i};

    %% Get Condition Matrices:
    task_blocks_dir = dir([subject_dir filesep 'Task_block*']);
    
    % Iterate through the multiple blocks:
    MRI_volume_vect = cell(1,size(task_blocks_dir,1));
    MRI_start_BLOCKIDX_vect = cell(1,size(task_blocks_dir,1));
    MRI_end_BLOCKIDX_vect = cell(1,size(task_blocks_dir,1));
    block_onset_vect = cell(1,size(task_blocks_dir,1));
    block_duration_vect = cell(1,size(task_blocks_dir,1));
    block_condition_vect = cell(1,size(task_blocks_dir,1));
    block_condition_onset_vect = cell(1,size(task_blocks_dir,1));
    block_condition_duration_vect = cell(1,size(task_blocks_dir,1));
    Exp_blocks_vect = cell(1,size(task_blocks_dir,1));
    block_BLOCKIDX_vect = cell(1,size(task_blocks_dir,1));
    class_MARKERS_vect = cell(1,size(task_blocks_dir,1));
    class_MARKERS_BLOCKIDX_vect = cell(1,size(task_blocks_dir,1));
    question_RESP_vect = cell(1,size(task_blocks_dir,1));
    
    for j = 1:size(task_blocks_dir,1)
        
        %%%%%%%%%%%%% Find the file with the most complete dataset %%%%%%%%%%%%%
        curr_block_dir = dir([subject_dir filesep task_blocks_dir(j).name filesep study_base_name curr_ID '_full_dataset*.mat']);
        if ~isempty(curr_block_dir) 
            load([curr_block_dir.folder, filesep, curr_block_dir.name]);
            
        else % The block did not reach completion - take the last saved file (last trial)
            curr_block_dir = dir([subject_dir filesep task_blocks_dir(j).name filesep study_base_name curr_ID '_block*.mat']);
            curr_block_dir_vect = cellfun(@(x) strsplit(x,{'_','.'}),{curr_block_dir.name},'UniformOutput',0);
            curr_block_dir_vect = cell2mat(cellfun(@(x)str2num(x{5}),curr_block_dir_vect,'UniformOutput',0));
            [~,curr_block_dir_vect_idx] = sort(curr_block_dir_vect); 
            
            Exp_blocks = load([curr_block_dir(curr_block_dir_vect_idx(1)).folder, filesep, curr_block_dir(curr_block_dir_vect_idx(1)).name],'Exp_blocks'); Exp_blocks = Exp_blocks.Exp_blocks;
            block_BLOCKIDX = cell(1,length(Exp_blocks));
            class_MARKERS = cell(1,length(Exp_blocks));
            class_MARKERS_BLOCKIDX = cell(1,length(Exp_blocks));
            question_RESP = cell(1,length(Exp_blocks));
            for k = 1:length(curr_block_dir_vect_idx)
                temp_load = load([curr_block_dir(curr_block_dir_vect_idx(k)).folder, filesep, curr_block_dir(curr_block_dir_vect_idx(k)).name]); 
                temp_idx = cellfun(@isempty,temp_load.block_BLOCKIDX,'UniformOutput',0); 
                block_BLOCKIDX{k} = temp_load.block_BLOCKIDX{~cell2mat(temp_idx)};
                class_MARKERS{k} = temp_load.class_MARKERS{~cell2mat(temp_idx)};
                class_MARKERS_BLOCKIDX{k} = temp_load.class_MARKERS_blockidx{~cell2mat(temp_idx)};
                question_RESP{k} = temp_load.question_RESP{~cell2mat(temp_idx)};               
            end
            save([subject_dir filesep task_blocks_dir(j).name filesep study_base_name curr_ID '_full_dataset1.mat'],...
                'Exp_blocks','block_BLOCKIDX','class_MARKERS','class_MARKERS_BLOCKIDX','question_RESP');
        end
        Exp_blocks_vect{j} = Exp_blocks;
        block_BLOCKIDX_vect{j} = block_BLOCKIDX;
        class_MARKERS_vect{j} = class_MARKERS;
        class_MARKERS_BLOCKIDX_vect{j} = class_MARKERS_BLOCKIDX;
        question_RESP_vect{j} = question_RESP;
        
        %%%%%%%%%%%%% Find the indices for each trial %%%%%%%%%%%%%
        curr_block_dir = dir([subject_dir filesep task_blocks_dir(j).name filesep study_base_name 'data_volume*.mat']);
        curr_block_dir_vect = cellfun(@(x) strsplit(x,{'volume','_','to','.'}),{curr_block_dir.name},'UniformOutput',0);
        [MRI_volume_vect{j},volume_sort_idx] = sort(cell2mat(cellfun(@(x)str2num(x{4}),curr_block_dir_vect,'UniformOutput',0))); 
        curr_block_dir_vect = curr_block_dir_vect(volume_sort_idx);
        MRI_start_BLOCKIDX_vect{j} = cell2mat(cellfun(@(x)str2num(x{5}),curr_block_dir_vect,'UniformOutput',0)); 
        MRI_end_BLOCKIDX_vect{j} = cell2mat(cellfun(@(x)str2num(x{6}),curr_block_dir_vect,'UniformOutput',0));        
        
        % Remove empty cells in block_BLOCKIDX_vect{j} - happens when the
        % participant does not finish all tasks within the time for one
        % block:
        empty_idx = cell2mat(cellfun(@isempty,block_BLOCKIDX_vect{j},'UniformOutput',0));
        Exp_blocks_vect{j} = Exp_blocks_vect{j}(~empty_idx);
        block_BLOCKIDX_vect{j} = block_BLOCKIDX_vect{j}(~empty_idx);
        class_MARKERS_vect{j} = class_MARKERS_vect{j}(~empty_idx);
        class_MARKERS_BLOCKIDX_vect{j} = class_MARKERS_BLOCKIDX_vect{j}(~empty_idx);
        question_RESP_vect{j} = question_RESP_vect{j}(~empty_idx);
        
        % Populate the vector indicating the onset and duration for each
        % trial (in seconds) from block_BLOCKIDX_vect that is in terms of 
        % EEG buffer blocks (correlated to MRI volumes by MRI_start_BLOCKIDX_vect):
        % Does EEG blocks to MRI volume conversion:
        block_onset_vect{j} = cellfun(@(x)find(diff(MRI_start_BLOCKIDX_vect{j} < x(1))) + 1,block_BLOCKIDX_vect{j},'UniformOutput',0);
        temp_idx = find(cell2mat(cellfun(@(x)isempty(x),block_onset_vect{j},'UniformOutput',0))); 
        if ~isempty(temp_idx) 
            for kk = 1:length(temp_idx) curr_temp_idx = temp_idx(kk); block_onset_vect{j}{curr_temp_idx} = 0; end 
        end
        block_onset_vect{j} = cell2mat(block_onset_vect{j})*TR; % Multiply by TR to convert to seconds
        
        block_duration_vect{j} = cellfun(@(x)find(diff(MRI_end_BLOCKIDX_vect{j} > x(2))) + 1,block_BLOCKIDX_vect{j},'UniformOutput',0);
        temp_idx = find(cell2mat(cellfun(@(x)isempty(x),block_duration_vect{j},'UniformOutput',0))); 
        if ~isempty(temp_idx) 
            for kk = 1:length(temp_idx) curr_temp_idx = temp_idx(kk); block_duration_vect{j}{curr_temp_idx} = 0; end 
        end
        block_duration_vect{j} = cell2mat(block_duration_vect{j})*TR - block_onset_vect{j}; % Multiply by TR to convert to seconds
        
        % Populate the vector indicating the condition for each trial
        % (1,2,3.. indexes correspond to the conditions in study_conditions):
        block_condition_vect{j} = zeros(length(study_conditions),length(Exp_blocks_vect{j}));
        block_condition_vect{j}(1,Exp_blocks_vect{j} == 0) = 1; % 0 is encoded as ABM
        block_condition_vect{j}(2,Exp_blocks_vect{j} == 1) = 2; % 1 is encoded as WM
        block_condition_vect{j}(4,find(diff(Exp_blocks_vect{j}) == 1)) = 4; % 1-0 is ABM-WM (start = lowernumber)
        block_condition_vect{j}(4,find(diff(Exp_blocks_vect{j}) == 1) + 1) = 5; % 1-0 is ABM-WM (end = highernumber)
        block_condition_vect{j}(5,find(diff(Exp_blocks_vect{j}) == -1)) = 6; % 0-1 is WM-ABM (start)
        block_condition_vect{j}(5,find(diff(Exp_blocks_vect{j}) == -1) + 1) = 7; % 0-1 is WM-ABM (end)
        
        block_condition_onset_vect{j} = cell(1,length(study_conditions));
        block_condition_duration_vect{j} = cell(1,length(study_conditions));
        for k = 1:length(study_conditions)
            curr_condition_vect = block_condition_vect{j}(k,:); min_val = min(curr_condition_vect(curr_condition_vect~=0));

            if (k >= 4) && (~isempty(min_val))
                min_condition_vect = find(curr_condition_vect==min_val);
                block_condition_onset_vect{j}{k} = block_onset_vect{j}(min_condition_vect);
                block_condition_duration_vect{j}{k} = block_duration_vect{j}(min_condition_vect) + block_duration_vect{j}(min_condition_vect + 1);                
            else
                block_condition_onset_vect{j}{k} = block_onset_vect{j}(block_condition_vect{j}(k,:) > 0);
                block_condition_duration_vect{j}{k} = block_duration_vect{j}(block_condition_vect{j}(k,:) > 0);                
            end
        end
        
        %%%%%%%%%%%%% Compile EEG for the trials in the current block %%%%%%%%%%%%%
        
        
        
    end
    
    save([subject_dir filesep curr_ID '_trial_data'])
    
    %% Check if MRI files already processed by checking anatomical directory:
    if (~exist([subject_dir, filesep, 'MRI', filesep, 'anat']))
        conn_precuratefMRI(subject_folder,scan_parameters,replace_files)
    end
end

%% Create Table:
sub_dir_table = table(subjectNumber_vect(exam_vect_dir_bin)',subjectID_vect(exam_vect_dir_bin)',sessionNumber_vect(exam_vect_dir_bin)',date_vect(exam_vect_dir_bin)',exam_vect',...
    'VariableNames',{'SubjectNumber', 'SubjectID', 'SessionNumber', 'Date', 'ExamNumber'});
writetable(sub_dir_table,[base_path_data filesep excel_name],'FileType','spreadsheet');

% Convert to JSON and write as .json file:
sub_dir_json = jsonencode(sub_dir_table);
fid = fopen([base_path_data filesep excel_name '.json'], 'w'); if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, sub_dir_json, 'char'); fclose(fid);