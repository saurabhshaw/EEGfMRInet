% Use this script to precurate the fMRI data for input into conn:
% clear all; clc;

function [] = conn_precuratefMRI(subject_folder,scan_parameters,replace_files)

% Control parameters:
% replace_files = 0;
% scan_parameters.anat_num_images = 184;
% scan_parameters.rsfunc_num_images = 5850;
% scan_parameters.tfunc_num_images = 11700;
% subject_folder = '20181218_CompositeTask_EEGfMRI-YC';

anat_num_images = scan_parameters.anat_num_images;
rsfunc_num_images = scan_parameters.rsfunc_num_images;
tfunc_num_images = scan_parameters.tfunc_num_images;

% Set Paths:
% [base_path_rc, base_path_rd] = setPaths();
% addpath([base_path_rc filesep 'Toolboxes']);

%% Define directories:
subject_dir = subject_folder;
base_path_curr = fileparts(mfilename('fullpath')); cd(base_path_curr); cd ..; base_path = pwd;
% subject_dir = [base_path_rd filesep subject_folder];
% conn_data_dir = [base_path_rd filesep 'Analyzed_data' filesep 'Composite_Task_EEG_fMRI'];
% dcm2nii_dir = [base_path_rc filesep 'Toolboxes' filesep 'dicm2nii']; % might need this - verify
dcm2nii_path = [base_path filesep 'Toolboxes' filesep 'Mricron'];

% addpath(genpath(dcm2nii_dir)); % might need this - verify
% addpath(genpath(dcm2nii_path));

%% Open up MRI images:
cd([subject_dir filesep 'MRI']);
curr_files = dir('exam*');
% curr_exam = strsplit(curr_files(2).name,{'_','.'}); curr_exam = str2num(curr_exam{2}); - Gave Index exceeds matrix dimensions error
curr_exam = strsplit(curr_files(1).name,{'_','.'}); curr_exam = str2num(curr_exam{2});

% Unzip if not already unzipped and move into folder:
if ~sum([curr_files(:).isdir]) & isempty(dir([subject_dir filesep 'MRI' filesep 'anat' filesep num2str(curr_exam) '_anat.nii']))      
    untar(curr_files(1).name);
end
[~,curr_dir,~] = fileparts(curr_files(1).name); cd(curr_dir);

% Identify series information:
% Read in seriesINFO text file:
curr_files = dir('seriesINFO*');
FID = fopen(curr_files(1).name);
formatSpec = '%s'; % formatSpec = '%s %d %s';
C = textscan(FID,formatSpec,'HeaderLines',1);
fclose(FID);
% Format read in data:
C = C{1};
unique_ser = find(not(cellfun('isempty',strfind(C,'series:')))); unique_ser = [unique_ser; length(C)];
series_INFO = cell(size(unique_ser)-1);
for i = 1:(length(unique_ser)-1)
    series_INFO{i} = strsplit([C{(unique_ser(i) + 2):(unique_ser(i+1)-1)}],{'(',')*'}); % Split at the brackets
    series_INFO{i}{2} = strsplit(series_INFO{i}{2},'images)'); % Convert the number of images into integer
    series_INFO{i}{2} = str2double(series_INFO{i}{2}{1});
end    

%% Identify the anatomical, resting fMRI and task fMRI Series:

% This should get the series number of the following scans:
curr_anat = find(cell2mat(cellfun(@(x)~isempty(strfind(x{1},'FSPGR'))&(x{2} == anat_num_images),series_INFO,'UniformOutput',false))); if length(curr_anat)>1 curr_anat = curr_anat(end); end
curr_rsfunc = find(cell2mat(cellfun(@(x)~isempty(strfind(x{1},'RESTING'))&(x{2} == rsfunc_num_images),series_INFO,'UniformOutput',false)));
curr_tfunc = find(cell2mat(cellfun(@(x)~isempty(strfind(x{1},'Task'))&(x{2} == tfunc_num_images),series_INFO,'UniformOutput',false))); 
% this is a list of length = number of blocks run

%% Convert DICOM to NIFTI files:
cd([subject_dir filesep 'MRI']);
if ~exist('anat') mkdir('anat'); end  % Create anat directory if does not exist
if ~exist('rsfunc') mkdir('rsfunc'); end  % Create func directory if does not exist
if ~exist('tfunc') mkdir('tfunc'); end  % Create func directory if does not exist

if replace_files
    cd('anat'); delete('*'); cd([subject_dir filesep 'MRI']);
    cd('rsfunc'); delete('*'); cd([subject_dir filesep 'MRI']);
    cd('tfunc'); delete('*');
end

% Check if already processed:
% Check anatomical directory:
cd(strcat(subject_dir,filesep, 'MRI', filesep, 'anat'));
exist_anat = exist([num2str(curr_exam), '_anat' ,'.nii.gz']);
% Check resting functional directory:
cd(strcat(subject_dir,filesep, 'MRI', filesep, 'rsfunc'));
exist_rsfunc = exist([num2str(curr_exam), '_rsfunc' ,'.nii.gz']);
% Check task functional directory:
cd(strcat(subject_dir,filesep, 'MRI', filesep, 'tfunc'));
exist_tfunc = exist([num2str(curr_exam), '_tfunc_block1' ,'.nii.gz']);
for i = 2:length(curr_tfunc)    
    exist_tfunc = exist_tfunc || exist([num2str(curr_exam), ['_tfunc_block' num2str(i)] ,'.nii.gz']);
end

% Convert the anatomical/functional data:
cd(dcm2nii_path);
if ~exist_anat
    dos(['dcm2nii.exe -4 y -o ', subject_dir, filesep, 'MRI', filesep,'anat',...
        ' -v y ', subject_dir, filesep, 'MRI',filesep,'exam_',num2str(curr_exam),...
        filesep,'Ser',num2str(curr_anat)]);
    
    cd(strcat(subject_dir, filesep, 'MRI', filesep, 'anat'));
    anat_name = dir('20*');
    copyfile(anat_name.name,[num2str(curr_exam), '_anat' ,'.nii.gz']);
    delete('20*.nii*','c*.nii*','o*.nii*');
end


cd(dcm2nii_path);
if ~exist_rsfunc
    dos(['dcm2nii.exe -4 y -o ', subject_dir, filesep, 'MRI',filesep,'rsfunc',...
        ' -v y ', subject_dir, filesep, 'MRI',filesep,'exam_',num2str(curr_exam),...
        filesep,'Ser',num2str(curr_rsfunc)]);
    
    cd(strcat(subject_dir, filesep, 'MRI', filesep, 'rsfunc'));
    rsfunc_name = dir('20*');
    copyfile(rsfunc_name.name,[num2str(curr_exam), '_rsfunc' ,'.nii.gz']);
    delete('20*.nii*','c*.nii*','o*.nii*');
end

if ~exist_tfunc
    for i = 1:length(curr_tfunc)
        cd(dcm2nii_path);
        dos(['dcm2nii.exe -4 y -o ', subject_dir, filesep, 'MRI',filesep,'tfunc',...
            ' -v y ', subject_dir, filesep, 'MRI',filesep,'exam_',num2str(curr_exam),...
            filesep,'Ser',num2str(curr_tfunc(i))]);
        
        cd(strcat(subject_dir, filesep, 'MRI', filesep, 'tfunc'));
        rsfunc_name = dir('20*');
        copyfile(rsfunc_name.name,[num2str(curr_exam), ['_tfunc_block' num2str(i)] ,'.nii.gz']);
        delete('20*.nii*','c*.nii*','o*.nii*');
    end
end

% A = dicm2nii(pwd, pwd, 'nii'); % Doesnt work yet!!