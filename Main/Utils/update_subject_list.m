function [sub_dir, sub_dir_mod] = update_subject_list(study_name,modality,base_path_data,runs_to_include)
%%
file_prefix = 'Subject_Processing_Info_';

% Get directory information:
sub_dir = dir([base_path_data filesep '*_' study_name '_' modality '*']);
sub_dir_mod = rmfield(sub_dir,{'folder','date','bytes','isdir','datenum'});

% Get processing information:
sub_dir_cell = {sub_dir.name};
temp_PID = cellfun(@(x) strsplit(x,{'-','_'}),sub_dir_cell,'un',0); temp_PID = cellfun(@(x) x{4},temp_PID,'un',0); [sub_dir_mod.PID] = temp_PID{:};
temp_SID = cellfun(@(x) strsplit(x,{'_','.mat'}),sub_dir_cell,'un',0); 
if length(temp_SID{1}) >= 4 % When they are multiple session studies
    temp_SID = cellfun(@(x) x{4},temp_SID,'un',0); 
else % When they are single session studies
    temp_SID = cellfun(@(x) '',temp_SID,'un',0);
end    
[sub_dir_mod.SID] = temp_SID{:};

for k = 1:length(runs_to_include)
    curr_run = runs_to_include{k};
    temp_isPreProcessed = cellfun(@(x,y,z) ~isempty(dir([base_path_data filesep x filesep 'PreProcessed' filesep curr_run '_' y '_' z '_preprocessed.set'])),sub_dir_cell,temp_PID,temp_SID,'un',0); 
    eval(['[sub_dir_mod.' curr_run '_isPreProcessed] = temp_isPreProcessed{:};']);
    temp_StageCompletion_dir = cellfun(@(x,y,z) dir([base_path_data filesep x filesep 'PreProcessed' filesep curr_run '_' y '_' z '_StageCompletion.mat']),sub_dir_cell,temp_PID,temp_SID,'un',0);
    temp_isStageCompletion = cellfun(@(x) ~isempty(x),temp_StageCompletion_dir);
    temp_StageCompletion = cell(size(temp_isStageCompletion)); temp_StageCompletion = cellfun(@(x) 0,temp_StageCompletion,'un',0);
    temp_StageCompletion(temp_isStageCompletion) = cellfun(@(x) load([x.folder filesep x.name],'max_finishedStage') ,temp_StageCompletion_dir(temp_isStageCompletion),'un',0);
    temp_StageCompletion(temp_isStageCompletion) = cellfun(@(x) x.max_finishedStage,temp_StageCompletion(temp_isStageCompletion),'un',0);
    eval(['[sub_dir_mod.' curr_run '_max_finishedStage] = temp_StageCompletion{:};']);
end

% Convert to table and write table as excel file:
sub_dir_table = struct2table(sub_dir_mod); writetable(sub_dir_table,[base_path_data filesep file_prefix study_name],'FileType','spreadsheet');

% Convert to JSON and write as .json file:
sub_dir_json = jsonencode(sub_dir_table);
fid = fopen([base_path_data filesep file_prefix study_name '.json'], 'w'); if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, sub_dir_json, 'char'); fclose(fid);