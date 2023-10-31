toolboxes_path = [base_path filesep 'Toolboxes']; base_path_main = [base_path filesep 'Main'];
eeglab_directory = [toolboxes_path filesep 'EEGLAB'];
addpath(genpath(base_path)); rmpath(genpath(toolboxes_path));
addpath(genpath(eeglab_directory));
addpath(genpath([toolboxes_path filesep 'libsvm-3.23'])); % Adding LibSVM
addpath(genpath([toolboxes_path filesep 'FSLib_v4.2_2017'])); % Adding FSLib