% Run this to set the paths to Research_code and Research_data at the
% beginning of every code:

function [base_path_rc, base_path_rd] = setPaths()

% Set paths:
[~,host_name] = system('hostname'); host_name = strtrim(host_name);
if strfind(host_name,'gra') host_name = 'graham'; end
switch host_name        
    case 'Saurabh-Laptop' % Saurabh Dell Laptop:
        base_path_rc = 'C:\Saurabh_files\Research_code'; % 'Research_code' folder location
        base_path_rd = 'C:\Saurabh_files\Research_data'; % 'Research_data' folder location when working from external hard drive
        % base_path_rd = 'C:\Saurabh_files\Research_data'; % 'Research_data' folder location
        
    case 'SAURABH-DESKTOP' % Saurabh Dell Laptop:
        base_path_rc = 'F:\Saurabh_files\Research_code'; % 'Research_code' folder location
        base_path_rd = 'F:\Saurabh_files\Research_data'; % 'Research_data' folder location when working from external hard drive
        % base_path_rd = 'C:\Saurabh_files\Research_data'; % 'Research_data' folder location
        
    case 'Sanjay-PC' % Home Desktop:
        base_path_rc = 'E:\Research\research_code'; % 'Research_code' folder location
        base_path_rd = 'E:\Research\research_data'; % 'Research_data' folder location
        
    case 'hypatia'
        base_path_rc = 'Z:\research_code'; % 'Research_code' folder location
        base_path_rd = 'Z:\Research_data'; % 'Research_data' folder location
        
    case 'hopper'
        base_path_rc = '/home/shaws5/research_code'; % 'Research_code' folder location
        base_path_rd = '/home/shaws5/Research_data'; % 'Research_data' folder location
        
    case 'MSI'
        base_path_rc = 'D:\Saurabh_files\Research_code'; % 'Research_code' folder location
        base_path_rd = 'D:\Saurabh_files\Research_data'; % 'Research_data' folder location when working from external hard drive
       
    case 'fossey'
        base_path_rc = '/home/shaws5/research_code'; % 'Research_code' folder location
        base_path_rd = '/home/shaws5/Research_data'; % 'Research_data' folder location
        
    case 'iqaluk.sharcnet.ca'
        base_path_rc = '/work/shaws5/Research_code'; % 'Research_code' folder location
        base_path_rd = '/work/shaws5/Research_data'; % 'Research_data' folder location
        
    case 'wobbie.sharcnet.ca'
        base_path_rc = '/work/shaws5/Research_code'; % 'Research_code' folder location
        base_path_rd = '/work/shaws5/Research_data'; % 'Research_data' folder location
		
	case 'graham' 
		base_path_rc = '/home/shaws5/projects/def-beckers/shaws5/Research_code'; % 'Research_code' folder location
        base_path_rd = '/home/shaws5/projects/def-beckers/shaws5/Research_data'; % 'Research_data' folder location
        
    case 'DESKTOP-N566C1R' % Tyler's Laptop
        base_path_rc = 'C:\Users\tyler\OneDrive - McMaster University\Documents\2018-2019\EEGfMRII'; % 'Research_code' folder location
        base_path_rd = 'C:\Users\tyler\OneDrive - McMaster University\Documents\2018-2019\EEGfMRII\MRMR Input Cropped'; % 'Research_data' folder location when working from external hard drive
end
% data_on_external_hdd = 0;

% if data_on_external_hdd
%     % drives = upper(getdrives);
%     base_path_rc = 'F:\Saurabh_files\Research_code'; % 'Research_code' folder location
%     base_path_rd = 'F:\Saurabh_files\Research_data'; % 'Research_data' folder location when working from external hard drive
% end

