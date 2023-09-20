% This function creates a new file for the epochs specified
function function_name = create_Mfile(base_path,input_filename,sub_dir_mod,sub_dir,datafile,input_epochs,subject_number,session_number)

function_name = [input_filename '_Subject' sub_dir_mod(subject_number).PID '_Session' sub_dir_mod(session_number).SID '_epoch' num2str(input_epochs(1)) 'to' num2str(input_epochs(end))];

% Read in the template function:
fid = fopen([base_path filesep 'Main' filesep input_filename '.m']);
file_data = []; i = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    file_data{i} = (tline);
    i = i + 1;
end
fclose(fid);

% Modify the lines that correspond to the inputs:
file_data{3} = ['datafile = ''' datafile ''';'];
file_data{4} = ['runepochs = ' mat2str(input_epochs) ';'];
file_data{5} = ['subject_number = ' mat2str(subject_number) ';'];
file_data{6} = ['session_number = ' mat2str(session_number) ';'];

% Write out the new .m file:
fout = fopen([base_path filesep 'Sharcnet' filesep 'sub_files' filesep function_name '.m'],'w');
for j = 1:length(file_data)
    fprintf(fout,'%s\n',file_data{j});    
end
fclose(fout);