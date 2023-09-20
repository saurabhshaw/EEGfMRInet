
input_file = [base_path filesep 'Main' filesep 'process_' study_name '.m'];
original_loopIDX_range = {[19:98]};
%original_loopIDX_range = {[1:length(Y_unique)] [1:length(Y_unique)]};
%original_loopIDX_range = {[2:length(sub_dir)] [1:5]};
% original_loopIDX_range = {[2:15] [1:4]};
%loopIDX_step = [1 1];
loopIDX_step = [1];

run_parfor = 0;
run_jobArray = 0;
name_suffix = '_NET';

graham_cfg = []; 
graham_cfg.req_cores = 16; % maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham. 32
graham_cfg.req_RAM = 32; % memory per node in GB 80
graham_cfg.req_queue='threaded';
graham_cfg.req_runtime='0-6:50'; % time (DD-HH:MM)
graham_cfg.account = 'rrg-beckers'; % def-beckers or rrg-beckers
graham_cfg.jobArray = '1-15%1'; % The number of jobs to run sequentially in job array: "start-end%step"
% Runtime for each job within job array - Keep in mind there are partitions for 
% 3 hours or less,
% 12 hours or less,
% 24 hours (1 day) or less,
% 72 hours (3 days) or less,
% 7 days or less, and
% 28 days or less
graham_cfg.jobArray_runtime = '0-02:50'; % time (DD-HH:MM)
    
graham_jobstorage_location = '''/tmp/jobstorage'''; % Location for each job to store its parallel job files

individual_template_file = 'Sharcnet_EEGnet_submit_template_2019b.sh'; % Change this if needed

%%
% function graham_parfor(input_file,base_path,original_loopIDX_range,loopIDX_step,run_parfor,graham_cfg)
% In the file that contains the for loop - add comment GRAHAM_PARFOR at the
% line containing the "for" statement and GRAHAM_PARFOR_END at the line containing the
% corresponding "end"
%
% If there are multiple for loops - original_loopIDX_range & loopIDX_step will be a cellarray with the ranges for each
%

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
fid = fopen([base_path filesep 'Sharcnet' filesep individual_template_file]);
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

% Find the number of unique parfor loops:
file_data_parfor_list = arrayfun(@(x) strsplit(file_data{x},'GRAHAM_PARFOR-'),file_data_parfor_idx,'un',0); file_data_parfor_list = cellfun(@(x) str2num(x{2}),file_data_parfor_list);
[file_data_parfor_unique,~,file_data_parfor_unique_idx] = unique(file_data_parfor_list); % THis is so that - file_data_parfor_unique(file_data_parfor_unique_idx) = file_data_parfor_list 
file_data_parfor_unique_cell = arrayfun(@(y) find(y == file_data_parfor_unique_idx),file_data_parfor_unique,'un',0); % Find the file_data_parfor_idx index that correspond to each of the file_data_parfor_unique
file_data_parfor_idx_cell = cellfun(@(x)file_data_parfor_idx(x),file_data_parfor_unique_cell,'un',0);
file_data_loop_variable = cellfun(@(x)strsplit(file_data{x(1)},{'for','='}),file_data_parfor_idx_cell,'un',0); file_data_loop_variable = cellfun(@(x)x{2},file_data_loop_variable,'un',0);

% Identify the line corresponding to getting filepath:
file_data_filepath_idx = find(cellfun(@(x) ~isempty(strfind(x,'fileparts(matlab.desktop.editor.getActiveFilename)')),file_data));

% Identify the line corresponding to creating jobstorage location:
file_data_jobstorage_idx = find(cellfun(@(x) ~isempty(strfind(x,'GRAHAM_JOBSTORAGE_LOCATION')),file_data));

% Compute the loopIDX ranges for each file:
if length(original_loopIDX_range) == length(file_data_parfor_unique)
    eval_code = ''; start_idx = []; end_idx = [];
    for i = 1:length(original_loopIDX_range)
        eval_code = [eval_code 'for ' file_data_loop_variable{i} ' = 1:' num2str(loopIDX_step(i)) ':' num2str(length(original_loopIDX_range{i})) ';' newline];
    end
    % eval_code = [eval_code 'if i < j;' newline];
    eval_code = [eval_code 'curr_start = cellfun(@(x)evalin(''base'',x),file_data_loop_variable)'' ; start_idx = [start_idx curr_start]; curr_end = min((curr_start + loopIDX_step'' - 1),cellfun(@(x)length(x),original_loopIDX_range)''); end_idx = [end_idx curr_end];'];
    % eval_code = [eval_code 'end;' newline];
    for i = 1:length(original_loopIDX_range) eval_code = [eval_code 'end;' newline]; end
    
else
    error('Not enough entries provided for original_loopIDX_range - It should equal the number of parfor loops being modified (X) : GRAHAM_PARFOR-X');
end

eval(eval_code);
% start_idx{k} = 1:loopIDX_step(i):length(original_loopIDX_range{i});
% end_idx{k} = min((start_idx + loopIDX_step(i) - 1),length(original_loopIDX_range{i}));


% Add code here that tests one run of the function and identifies the
% runtime, using that information to generate the .sh file requirements

% Create output directories if not already made:
if ~isdir([base_path filesep 'Sharcnet' filesep 'sub_files']) mkdir([base_path filesep 'Sharcnet' filesep 'sub_files']); end
if ~isdir([base_path filesep 'Sharcnet' filesep 'out_files']) mkdir([base_path filesep 'Sharcnet' filesep 'out_files']); end
if ~isdir([base_path filesep 'Sharcnet' filesep 'sh_files']) mkdir([base_path filesep 'Sharcnet' filesep 'sh_files']); end

% Generate a new .m file and .sh file for each loopIDX range identified:
for i = 1:size(start_idx,2)
    if run_parfor for_text = 'parfor '; else for_text = 'for '; end
    
    curr_start_idx = zeros(1,size(start_idx,1)); curr_end_idx = zeros(1,size(start_idx,1));
    for m = 1:size(start_idx,1)
        curr_start_idx(m) = original_loopIDX_range{m}(start_idx(m,i)); curr_end_idx(m) = original_loopIDX_range{m}(end_idx(m,i));
    end
    
    % Modify the "for" loop statement:
    for m = 1:size(start_idx,1)
        for n = 1:length(file_data_parfor_idx_cell{m})
            file_data{file_data_parfor_idx_cell{m}(n)} = [for_text file_data_loop_variable{m} ' = ' num2str(curr_start_idx(m)) ':' num2str(curr_end_idx(m))];
        end
    end
    
    % Modify the "end" statements for the "for" loop:
    for kk = 1:length(file_data_parforend_idx)
        file_data{file_data_parforend_idx(kk)} = 'end % GRAHAM_PARFOR_END';
    end
    
    % Modify the output path statement:
    file_data{file_data_outputPath_idx} = ['output_base_path_data = ''' base_path_output_sharcnet '''; %GRAHAM_OUTPUT_PATH'];
    
    % Modify the filepath statement, adding another cd .. since this file will be nested in two levels of folders:
    file_data{file_data_filepath_idx} = 'base_path_main = fileparts(mfilename(''fullpath'')); cd(base_path_main); cd ..; cd ..;';
    % file_data{file_data_filepath_idx} = [file_data{file_data_filepath_idx} ' cd ..;'];
        
    % Make the jobstorage location file:
    file_data{file_data_jobstorage_idx} = ['mkdir(' graham_jobstorage_location '); %GRAHAM_JOBSTORAGE_LOCATION'];
    
    % Write out the new .m file:
    function_name = [input_filename '_loopIDX_' regexprep(mat2str(curr_start_idx),{'[',']',' '},{'','','_'}) 'to' regexprep(mat2str(curr_end_idx),{'[',']',' '},{'','','_'}) name_suffix];
    fout = fopen([base_path filesep 'Sharcnet' filesep 'sub_files' filesep function_name '.m'],'w');
    for j = 1:length(file_data)
        fprintf(fout,'%s\n',file_data{j});
    end
    fclose(fout);
    
    % Modify the .sh file structure:
    graham_cfg.out_name = [base_path_rc_sharcnet '/EEGnet/Sharcnet/out_files/' function_name '-Node%N-JobID%j.out'];
    sh_file_data{2} = ['#SBATCH --cpus-per-task=' num2str(graham_cfg.req_cores) '   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.'];
    sh_file_data{3} = ['#SBATCH --mem=' num2str(graham_cfg.req_RAM) 'G        # memory per node'];
    if ~run_jobArray 
        sh_file_data{4} = ['#SBATCH --time=' graham_cfg.req_runtime '      # time (DD-HH:MM)'];
    else
        sh_file_data{4} = ['#SBATCH --time=' graham_cfg.jobArray_runtime '      # time (DD-HH:MM)'];
    end
    sh_file_data{5} = ['#SBATCH --output=' graham_cfg.out_name '  # %N for node name, %j for jobID'];
    sh_file_data{6} = ['#SBATCH --account=' graham_cfg.account];
    if run_jobArray sh_file_data{7} = ['#SBATCH --array=' graham_cfg.jobArray '    # Run a 10-job array, one job at a time.']; end
    sh_file_data{9} = ['cd ' base_path_rc_sharcnet '/EEGnet/Sharcnet/sub_files'];
    sh_file_data{10} = ['matlab -nodesktop -nosplash -nodisplay -r "run(''' function_name '.m' '''); exit"'];
    
    % Write out the .sh files:  
    function_name = [input_filename '_loopIDX_' regexprep(mat2str(curr_start_idx),{'[',']',' '},{'','','_'}) 'to' regexprep(mat2str(curr_end_idx),{'[',']',' '},{'','','_'}) name_suffix];
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
temp_shAll_file_data = shAll_file_data; index_var_offset = size(start_idx,1);
loopIdx_string = strsplit(mat2str((1:size(start_idx,2))-1),{'[',']'}); loopIdx_string = loopIdx_string{2};
% loopIdx_string = strsplit(mat2str(original_loopIDX_range(start_idx)),{'[',']'}); loopIdx_string = loopIdx_string{2};
shAll_file_data{8} = ['base_path_rc="' base_path_rc_sharcnet '"'];
shAll_file_data{9} = ['sh_file_prefix="' input_filename '_loopIDX_"'];
shAll_file_data{10} = ['loopIDX_step=' num2str(loopIDX_step)];
for m = 1:index_var_offset
    curr_str = regexprep(mat2str(original_loopIDX_range{m}(start_idx(m,:))),'[','('); curr_str = regexprep(curr_str,']',')'); 
    shAll_file_data{10+(2*(m-1))} = ['loopIDX_start' num2str(m) '=' curr_str];
    
    curr_str = regexprep(mat2str(original_loopIDX_range{m}(end_idx(m,:))),'[','('); curr_str = regexprep(curr_str,']',')');
    shAll_file_data{10+(2*(m-1))+1} = ['loopIDX_end' num2str(m) '=' curr_str];
end
shAll_file_data{10+(2*index_var_offset)+1} = ['for Idx in ' loopIdx_string];
shAll_file_data{10+(2*index_var_offset)+2} = ['do'];

startIdx_string = '    startIDX='; endIdx_string = '    endIDX=';
for m = 1:index_var_offset
    startIdx_string = [startIdx_string '${loopIDX_start' num2str(m) '[Idx]}']; if m~=index_var_offset startIdx_string = [startIdx_string '_'];end
    endIdx_string = [endIdx_string '${loopIDX_end' num2str(m) '[Idx]}']; if m~=index_var_offset endIdx_string = [endIdx_string '_'];end
end
shAll_file_data{10+(2*index_var_offset)+3} = startIdx_string;
shAll_file_data{10+(2*index_var_offset)+4} = endIdx_string;
shAll_file_data{10+(2*index_var_offset)+5} = ['curr_file=' input_filename '_loopIDX_${startIDX}to${endIDX}' name_suffix];
shAll_file_data(10+(2*index_var_offset)+6:10+(2*index_var_offset)+6+length(temp_shAll_file_data)-18) = temp_shAll_file_data(18:end);


% Write out the final .sh files:  
function_name = ['SUBMIT_' input_filename '_loopIDX' name_suffix];
fout = fopen([base_path filesep 'Sharcnet' filesep function_name '.sh'],'w');
for j = 1:length(shAll_file_data)
    fprintf(fout,'%s\n',shAll_file_data{j});
end
fclose(fout);