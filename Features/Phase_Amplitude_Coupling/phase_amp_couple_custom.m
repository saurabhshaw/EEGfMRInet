function [arr_sortamp, arr_plotamp, arr_avgamp] = phase_amp_couple_custom(dataset, window_length, window_step, PAC_opt, par_window)
[start_idx, end_idx] = create_windows(size(dataset{1},2), window_step, window_length); % Define Windowing

arr_sortamp = cell(1,length(start_idx));
arr_plotamp = cell(1,length(start_idx));
arr_avgamp = cell(1,length(start_idx));

if par_window    
    parfor j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [arr_sortamp{j}, arr_plotamp{j}, arr_avgamp{j}] = phase_amp_couple_custom_freq(curr_dataset, PAC_opt);
    end
    
else    
    for j = 1:length(start_idx) % Iterate through the windows
        curr_dataset = cellfun(@(x)squeeze(x(:,start_idx(j):end_idx(j))),dataset,'UniformOutput',0);
        [arr_sortamp{j}, arr_plotamp{j}, arr_avgamp{j}] = phase_amp_couple_custom_freq(curr_dataset, PAC_opt);
    end
end


end

