function [CFC_SI, CFC_SI_mag, CFC_SI_theta] = CFC_SynchronizationIndex(dataset, window_length, window_step, CFCSI_opt, par_window)

[start_idx, end_idx] = create_windows(size(dataset{1},2), window_step, window_length); % Define Windowing

CFC_SI = cell(1,length(start_idx));
CFC_SI_mag = cell(1,length(start_idx));
CFC_SI_theta = cell(1,length(start_idx));

if par_window    
    parfor j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [CFC_SI{j}, CFC_SI_mag{j}, CFC_SI_theta{j}] = compute_CFCSI_freq(curr_dataset, CFCSI_opt);
    end
    
else    
    for j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [CFC_SI{j}, CFC_SI_mag{j}, CFC_SI_theta{j}] = compute_CFCSI_freq(curr_dataset, CFCSI_opt);
    end
end

end



