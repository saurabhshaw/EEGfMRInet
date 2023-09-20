function delete_PreProcessed(study_name,modality,base_path_data)

% Get directory information:
sub_dir = dir([base_path_data filesep '*_' study_name '_' modality '*']); sub_dir_cell = {sub_dir.name};
temp_isPreProcessed = cellfun(@(x) ~isempty(dir([base_path_data filesep x filesep 'PreProcessed'])),sub_dir_cell);
cellfun(@(x) rmdir([base_path_data filesep x filesep 'PreProcessed'],'s'),sub_dir_cell(temp_isPreProcessed));