function EEG = offline_preprocess_manual_compiled(cfg,curr_dir,dataset_name,overwrite_files,EEG,base_path)

temp_folder = cfg.temp_file; if isempty(dir(temp_folder)) mkdir(temp_folder); end
% temp_folder = [base_path filesep 'temp_deploy_files'];
temp_name = tempname(temp_folder);

% Write out file:
save(temp_name);

% Call the function:
if ismac
    % Code to run on Mac platform
elseif isunix
    %[status,results] = system([base_path filesep 'PreProcessing' filesep 'offline_preprocess_manual_deploy ' temp_name '.mat'],'-echo');

elseif ispc
    [status,results] = system([base_path filesep 'PreProcessing' filesep 'offline_preprocess_manual_deploy.exe ' temp_name '.mat'],'-echo');
    % EEG = offline_preprocess_manual_deploy_mod(temp_name);
end
EEG = offline_preprocess_manual_deploy([temp_name '.mat']);

% Clean up written out files:
delete([temp_name '.mat']);