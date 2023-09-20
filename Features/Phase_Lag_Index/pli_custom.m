function [PLI, PLIcorr, PLI_z_score] = pli_custom(dataset, window_length, window_step, pli_opt, par_window)

[start_idx, end_idx] = create_windows(size(dataset{1},2), window_step, window_length); % Define Windowing

PLI = cell(1,length(start_idx));
PLIcorr = cell(1,length(start_idx));
PLI_z_score = cell(1,length(start_idx));

if par_window    
    parfor j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [PLI{j}, PLIcorr{j}, PLI_z_score{j}] = pli_function_custom(curr_dataset, pli_opt);
    end
    
else    
    for j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [PLI{j}, PLIcorr{j}, PLI_z_score{j}] = pli_function_custom(curr_dataset, pli_opt);
    end
end

end

