% Compute features from the preprocessed dataset
function compute_features_compiled(EEG,curr_dir,dataset_name,feature_names,base_path)

% temp_folder = [base_path filesep 'temp_deploy_files'];
temp_folder = [curr_dir filesep 'temp_deploy_files']; if isempty(dir(temp_folder)) mkdir(temp_folder); end
temp_name = tempname(temp_folder);

% Write out file:
save(temp_name);

% Call the function:
if ismac
    % Code to run on Mac platform
elseif isunix
    % [status,results] = system([base_path filesep 'Features' filesep 'Compute_features_scripts' filesep 'compute_features_deploy ' temp_name '.mat'],'-echo');

elseif ispc
    [status,results] = system([base_path filesep 'Features' filesep 'Compute_features_scripts' filesep 'compute_features_deploy.exe ' temp_name '.mat'],'-echo');
    % EEG = offline_preprocess_manual_deploy_mod(temp_name);
end
% [status,results] = system(['test_deploy.exe ' temp_name '.mat'],'-echo')
compute_features_deploy(temp_name)

% Clean up written out files:
delete([temp_name '.mat']);