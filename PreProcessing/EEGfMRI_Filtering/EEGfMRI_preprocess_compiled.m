function EEGfMRI_preprocess_compiled(EEG,scan_parameters,output_dir,fastr_param,overwrite_files,curr_dir,base_path)

% [EEG] = EEGfMRI_preprocess(EEG,scan_parameters,output_dir,fastr_param,overwrite_files)

% temp_folder = [base_path filesep 'temp_deploy_files'];
temp_folder = [curr_dir filesep 'temp_deploy_files']; if isempty(dir(temp_folder)) mkdir(temp_folder); end
temp_name = tempname(temp_folder);

% Write out file:
save(temp_name);

% Call the function:
if ismac
    % Code to run on Mac platform
elseif isunix
    [status,results] = system([base_path filesep 'PreProcessing' filesep 'EEGfMRI_Filtering' filesep 'EEGfMRI_preprocess_deploy ' temp_name '.mat'],'-echo');
    % EEGfMRI_preprocess_deploy(temp_name)
elseif ispc
    [status,results] = system([base_path filesep 'PreProcessing' filesep 'EEGfMRI_Filtering' filesep 'EEGfMRI_preprocess_deploy.exe ' temp_name '.mat'],'-echo');
    % EEG = offline_preprocess_manual_deploy_mod(temp_name);
end


% Clean up written out files:
delete([temp_name '.mat']);