function [dPLI, dPLIcorr, dPLI_z_score] = dpli_custom(dataset, window_length, window_step, dpli_opt, par_window)

[start_idx, end_idx] = create_windows(size(dataset{1},2), window_step, window_length); % Define Windowing

dPLI = cell(1,length(start_idx));
dPLIcorr = cell(1,length(start_idx));
dPLI_z_score = cell(1,length(start_idx));

if par_window    
    parfor j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [dPLI{j}, dPLIcorr{j}, dPLI_z_score{j}] = dpli_function_custom(curr_dataset, dpli_opt);
    end
    
else    
    for j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [dPLI{j}, dPLIcorr{j}, dPLI_z_score{j}] = dpli_function_custom(curr_dataset, dpli_opt);
    end
end

end

