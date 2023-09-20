base_path_rd = 'Z:\Research_data';
A_dir = dir([base_path_rd filesep '*_RunningNFB_*']);
A_dir_blocks = cell(1,length(A_dir)); A_dir_fulldataset = cell(1,length(A_dir));
for i = 1:length(A_dir)
    A_dir_blocks{i} = dir([A_dir(i).folder filesep A_dir(i).name filesep '*_block_*.mat']);  
    A_dir_fulldataset{i} = dir([A_dir(i).folder filesep A_dir(i).name filesep '*_full_dataset.mat']);
    
    if (~isempty(A_dir_fulldataset{i})) && (~isempty(A_dir_blocks{i}))
        for j = 1:length(A_dir_blocks{i})
            delete([A_dir_blocks{i}(j).folder filesep A_dir_blocks{i}(j).name]);
        end        
    end
end