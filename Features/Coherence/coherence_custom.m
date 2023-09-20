function [avg_coherence, all_coherence] = coherence_custom(dataset, window_length, window_step, srate, par_window)

[start_idx, end_idx] = create_windows(size(dataset{1},2), window_step, window_length); % Define Windowing

all_coherence = cell(1,length(start_idx));
avg_coherence = cell(1,length(start_idx));

if par_window    
    parfor j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [avg_coherence{j}, all_coherence{j}] = coherence_function_custom(curr_dataset, srate);
    end
else    
    for j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [avg_coherence{j}, all_coherence{j}] = coherence_function_custom(curr_dataset, srate);
    end    
end

end




















% tic
% [full_coherence{1}, avg_coherence{1}] = coherence_eegapp(dataset, start_idx(1), end_idx(1), 'full');
% toc


%dataset.data = dataset.data(:,start_idx(1):end_idx(1));
%[siz1,siz2] = coherence_eegapp(dataset,channels,channels,bandpass);
%avg_coherence = zeros(size(siz1,1),size(siz1,2),length(start_idx));
%full_coherence = zeros(size(siz2,1),size(siz2,2),size(siz2,3),length(start_idx));

%avg_coherence(:,:,1) = siz1;
%full_coherence(:,:,:,1) = siz2;

%     
%     [avg_coherence(:,:,j), full_coherence(:,:,:,j)] = coherence_eegapp(dataset,channels,channels,bandpass);
% %        [full_coherence{j}, avg_coherence{j}] = coherence_calc(dataset,mode,start_idx(j),end_idx(j)); 

% 
% for j = 1:length(start_idx)
% tic
%     dataset.data = mama_dataset.data(:,start_idx(j):end_idx(j));
%     [avg_coherence(:,:,j), full_coherence(:,:,:,j)] = coherence_eegapp(dataset,channels,channels,bandpass); 
% toc
% end    
