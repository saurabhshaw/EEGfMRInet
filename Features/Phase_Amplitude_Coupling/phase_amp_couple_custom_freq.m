function [arr_sortamp, arr_plotamp, arr_avgamp] = phase_amp_couple_custom_freq(curr_dataset, PAC_opt)

frequency_bands = PAC_opt.frequency_bands;
arr_sortamp = cell(length(frequency_bands),length(frequency_bands));
arr_plotamp = cell(length(frequency_bands),length(frequency_bands));
arr_avgamp = cell(length(frequency_bands),length(frequency_bands));

for i = 1:length(frequency_bands) % Low frequency band
    % low_band = bandWidth(frequency_bands{i});
    for j = 1:length(frequency_bands) % High frequency band
        % high_band = bandWidth(frequency_bands{j});
        if (i < j)            
            % higher_freq_low = high_band(1); higher_freq_high = high_band(2);
            % lower_freq_low = low_band(1); lower_freq_high = low_band(2);
            % CFC_SI{i,j} = compute_CFCSI(curr_dataset,higher_freq_low,higher_freq_high,lower_freq_low,lower_freq_high);
            if PAC_opt.bichannel
                [arr_plotamp{i,j}, arr_avgamp{i,j}] = phase_amplitude_coupling_function_custom_bichannel(curr_dataset,j,i,PAC_opt);
				arr_sortamp{i,j} = [];
            else
                [arr_sortamp{i,j}, arr_plotamp{i,j}, arr_avgamp{i,j}] = phase_amplitude_coupling_function_custom(curr_dataset,j,i,PAC_opt);
            end
        end
    end
end
