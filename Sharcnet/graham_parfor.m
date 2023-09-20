
input_file = [base_path filesep 'Main' filesep 'process_' study_name '.m'];
original_loopIDX_range = 1:144;
% original_loopIDX_range = 1:length(sub_dir);
% original_loopIDX_range = 1:length(start_idx);
% original_loopIDX_range = 1:length(Y_unique);
loopIDX_step = 1;
run_parfor = 0;

graham_cfg = []; 
graham_cfg.req_cores = 8; % maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
graham_cfg.req_RAM = 64; % memory per node in GB
graham_cfg.req_queue='threaded';
graham_cfg.req_runtime='0-02:50'; % time (DD-HH:MM)
graham_cfg.account = 'rrg-beckers'; % def-beckers or rrg-beckers

graham_jobstorage_location = '''/tmp/jobstorage'''; % Location for each job to store its parallel job files

%%
% function graham_parfor(input_file,base_path,original_loopIDX_range,loopIDX_step,run_parfor,graham_cfg)
% In the file that contains the for loop - add comment GRAHAM_PARFOR at the
% line containing the "for" statement and GRAHAM_PARFOR_END at the line containing the
% corresponding "end"

base_path_rc_sharcnet = '/home/shaws5/projects/def-beckers/shaws5/Research_code';
base_path_rd_sharcnet = '/home/shaws5/projects/def-beckers/shaws5/Research_data';
% base_path_output_sharcnet = '/home/shaws5/scratch/Research_data'; % This might be causing some errors with it not being accessible since it is going through a shortcut
% base_path_output_sharcnet = '/scratch/shaws5/Research_data'; 
base_path_output_sharcnet = '/project/rrg-beckers/shaws5/Research_data';


% Read in the template function:
[~,input_filename] = fileparts(input_file);
fid = fopen(input_file);
file_data = []; i = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    file_data{i} = (tline);
    i = i + 1;
end
fclose(fid);

% Read in the .sh template function:
fid = fopen([base_path filesep 'Sharcnet' filesep 'Sharcnet_EEGnet_submit_template.sh']);
sh_file_data = []; i = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    sh_file_data{i} = (tline);
    i = i + 1;
end
fclose(fid);

% Identify the lines that correspond to the "GRAHAM_PARFOR" tag:
file_data_parforend_idx = find(cellfun(@(x) ~isempty(strfind(x,'GRAHAM_PARFOR_END')),file_data)); % First find the END statement
file_data_parfor_idx = setdiff(find(cellfun(@(x) ~isempty(strfind(x,'GRAHAM_PARFOR')),file_data)),file_data_parforend_idx); % Find the FOR statement, removing the END statement
file_data_outputPath_idx = setdiff(find(cellfun(@(x) ~isempty(strfind(x,'GRAHAM_OUTPUT_PATH')),file_data)),file_data_parforend_idx); % Find the GRAHAM OUTPUT PATH statement
file_data_loop_variable = strsplit(file_data{file_data_parfor_idx},{'for','='}); file_data_loop_variable = file_data_loop_variable{2};

% Identify the line corresponding to getting filepath:
file_data_filepath_idx = find(cellfun(@(x) ~isempty(strfind(x,'fileparts(matlab.desktop.editor.getActiveFilename)')),file_data));

% Identify the line corresponding to creating jobstorage location:
file_data_jobstorage_idx = find(cellfun(@(x) ~isempty(strfind(x,'GRAHAM_JOBSTORAGE_LOCATION')),file_data));

% Compute the loopIDX ranges for each file:
start_idx = 1:loopIDX_step:length(original_loopIDX_range);
end_idx = min((start_idx + loopIDX_step - 1),length(original_loopIDX_range));

% Add code here that tests one run of the function and identifies the
% runtime, using that information to generate the .sh file requirements

% Create output directories if not already made:
if ~isdir([base_path filesep 'Sharcnet' filesep 'sub_files']) mkdir([base_path filesep 'Sharcnet' filesep 'sub_files']); end
if ~isdir([base_path filesep 'Sharcnet' filesep 'out_files']) mkdir([base_path filesep 'Sharcnet' filesep 'out_files']); end
if ~isdir([base_path filesep 'Sharcnet' filesep 'sh_files']) mkdir([base_path filesep 'Sharcnet' filesep 'sh_files']); end

% Generate a new .m file and .sh file for each loopIDX range identified:
for i = 1:length(start_idx)
    if run_parfor for_text = 'parfor '; else for_text = 'for '; end
    % if run_parfor for_text = 'parfor '; else for_text = ''; end

    curr_start_idx = original_loopIDX_range(start_idx(i)); curr_end_idx = original_loopIDX_range(end_idx(i));
    
    % Modify the "for" loop statement:
    file_data{file_data_parfor_idx} = [for_text file_data_loop_variable ' = ' num2str(curr_start_idx) ':' num2str(curr_end_idx)];
    
    % Modify the output path statement:
    file_data{file_data_outputPath_idx} = ['output_base_path_data = ''' base_path_output_sharcnet '''; %GRAHAM_OUTPUT_PATH'];
    
    % Modify the filepath statement, adding another cd .. since this file will be nested in two levels of folders:
    file_data{file_data_filepath_idx} = 'base_path_main = fileparts(mfilename(''fullpath'')); cd(base_path_main); cd ..; cd ..;';
    % file_data{file_data_filepath_idx} = [file_data{file_data_filepath_idx} ' cd ..;'];
    
    % Make the jobstorage location file:
    file_data{file_data_jobstorage_idx} = ['mkdir(' graham_jobstorage_location '); %GRAHAM_JOBSTORAGE_LOCATION'];
    
    % Write out the new .m file:    
    function_name = [input_filename '_loopIDX_' num2str(curr_start_idx) 'to' num2str(curr_end_idx)];
    fout = fopen([base_path filesep 'Sharcnet' filesep 'sub_files' filesep function_name '.m'],'w');
    for j = 1:length(file_data)
        fprintf(fout,'%s\n',file_data{j});
    end
    fclose(fout);
    
    % Modify the .sh file structure:
    graham_cfg.out_name = [base_path_rc_sharcnet '/EEGnet/Sharcnet/out_files/' function_name '-Node%N-JobID%j.out'];
    sh_file_data{2} = ['#SBATCH --cpus-per-task=' num2str(graham_cfg.req_cores) '   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.'];
    sh_file_data{3} = ['#SBATCH --mem=' num2str(graham_cfg.req_RAM) 'G        # memory per node'];
    sh_file_data{4} = ['#SBATCH --time=' graham_cfg.req_runtime '      # time (DD-HH:MM)'];
    sh_file_data{5} = ['#SBATCH --output=' graham_cfg.out_name '  # %N for node name, %j for jobID'];
    sh_file_data{6} = ['#SBATCH --account=' graham_cfg.account];
    sh_file_data{9} = ['cd ' base_path_rc_sharcnet '/EEGnet/Sharcnet/sub_files'];
    sh_file_data{10} = ['matlab -nodesktop -nosplash -nodisplay -r "run(''' function_name '.m' '''); exit"'];
    
    % Write out the .sh files:  
    function_name = [input_filename '_loopIDX_' num2str(curr_start_idx) 'to' num2str(curr_end_idx)];
    fout = fopen([base_path filesep 'Sharcnet' filesep 'sh_files' filesep function_name '.sh'],'w');
    for j = 1:length(sh_file_data)
        fprintf(fout,'%s\n',sh_file_data{j});
    end
    fclose(fout);
end

%% Create the final .sh file that will submit the remaining .sh files:
% Read in the final .sh template function:
fid = fopen([base_path filesep 'Sharcnet' filesep 'Sharcnet_EEGnet_submitAll_template.sh']);
shAll_file_data = []; i = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    shAll_file_data{i} = (tline);
    i = i + 1;
end
fclose(fid);

% Modify the final .sh file structure:
loopIdx_string = strsplit(mat2str(original_loopIDX_range(start_idx)),{'[',']'}); loopIdx_string = loopIdx_string{2};
shAll_file_data{8} = ['base_path_rc="' base_path_rc_sharcnet '"'];
shAll_file_data{9} = ['sh_file_prefix="' input_filename '_loopIDX_"'];
shAll_file_data{10} = ['loopIDX_step=' num2str(loopIDX_step)];
shAll_file_data{12} = ['for startIdx in ' loopIdx_string];

% Write out the final .sh files:  
function_name = ['SUBMIT_' input_filename '_loopIDX'];
fout = fopen([base_path filesep 'Sharcnet' filesep function_name '.sh'],'w');
for j = 1:length(shAll_file_data)
    fprintf(fout,'%s\n',shAll_file_data{j});
end
fclose(fout);