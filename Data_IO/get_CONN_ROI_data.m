% Define constants:
conn_folder = 'D:\Saurabh_files\Research_data\CONN_projects';
conn_project_name = 'HCWMIVO';
ROIs_toUse = {182, 183};
time_length = 556;
num_subs = 35;
use_All_cond = 1;

%% Load the CONN project settings:
load([conn_folder filesep conn_project_name '.mat']);
conditions = CONN_x.Setup.conditions.allnames;
if use_All_cond
    cond_All = find(cellfun(@(x)~isempty(x),strfind(conditions,'All')));
    if isempty(cond_All)
        use_All_cond = 0; % "All" condition does not exist in this dataset
        condIdx = 1;
    else
        condIdx = cond_All;
    end
end

%% Gather data from all participants:
data_path = [conn_folder filesep conn_project_name filesep 'results' filesep 'preprocessing'];
data_curated = zeros(length(ROIs_toUse),time_length,num_subs);
for i = 1:num_subs
    curr_data = load([data_path filesep 'ROI_Subject' sprintf('%03d',i) '_Condition' sprintf('%03d',condIdx) '.mat'],'data'); % Add if statement checking cond_All here
    for j = 1:size(data_curated)
        temp_X = cell2mat(curr_data.data(ROIs_toUse{j}));
        if size(temp_X,2) > 1 temp_X = mean(temp_X,2); end
        if isempty(temp_X) temp_X = nan(time_length,1); end
        if (length(temp_X) ~= time_length) pad_amt = time_length - length(temp_X); temp_X = [temp_X; zeros(pad_amt,1)]; end
        data_curated(j,:,i) = temp_X;
    end
end
