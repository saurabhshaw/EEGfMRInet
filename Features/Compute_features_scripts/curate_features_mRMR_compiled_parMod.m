% Compute features from the preprocessed dataset
function curate_features_mRMR_compiled_parMod(Featurefiles_basename, Featurefiles_directory, YY_final, max_features, curr_dir, base_path)

% temp_folder = [base_path filesep 'temp_deploy_files'];
temp_folder = [curr_dir filesep 'temp_deploy_files']; if isempty(dir(temp_folder)) mkdir(temp_folder); end
temp_name = tempname(temp_folder);

% Write out file:
save(temp_name);

% Call the function:
if ismac
    % Code to run on Mac platform
elseif isunix
    % [status,results] = system([base_path filesep 'Features' filesep 'Compute_features_scripts' filesep 'curate_features_mRMR_deploy_parMod ' temp_name '.mat'],'-echo');

elseif ispc
    [status,results] = system([base_path filesep 'Features' filesep 'Compute_features_scripts' filesep 'curate_features_mRMR_deploy_parMod.exe ' temp_name '.mat'],'-echo');
    % EEG = offline_preprocess_manual_deploy_mod(temp_name);
end
% [status,results] = system(['test_deploy.exe ' temp_name '.mat'],'-echo')
curate_features_mRMR_deploy_parMod(temp_name)

% Clean up written out files:
delete([temp_name '.mat']);