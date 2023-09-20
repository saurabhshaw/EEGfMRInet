function [CFC_SI, CFC_SI_mag, CFC_SI_theta] = compute_CFCSI_freq(curr_dataset, CFCSI_opt)

frequency_bands = CFCSI_opt.frequency_bands;
CFC_SI = cell(length(frequency_bands),length(frequency_bands));
CFC_SI_mag = cell(length(frequency_bands),length(frequency_bands));
CFC_SI_theta = cell(length(frequency_bands),length(frequency_bands));

for i = 1:length(frequency_bands) % Low frequency band
    % low_band = bandWidth(frequency_bands{i});
    for j = 1:length(frequency_bands) % High frequency band
        % high_band = bandWidth(frequency_bands{j});
        if (i < j)            
            % higher_freq_low = high_band(1); higher_freq_high = high_band(2);
            % lower_freq_low = low_band(1); lower_freq_high = low_band(2);
            % CFC_SI{i,j} = compute_CFCSI(curr_dataset,higher_freq_low,higher_freq_high,lower_freq_low,lower_freq_high);
            [CFC_SI{i,j}, CFC_SI_mag{i,j}, CFC_SI_theta{i,j}] = compute_CFCSI(curr_dataset,j,i);
        end
    end
end
