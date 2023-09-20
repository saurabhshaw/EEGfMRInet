%Looping_Code.m
%Ali Howidi, howidia, 400143756

frequency_bands = {'delta','theta','alpha','beta','gamma','high_gamma'};
subject_number = 1;
session_number = 1;


CFC_SI = cell(length(frequency_bands),length(frequency_bands));
for i = 1:length(frequency_bands)
    low_band = bandWidth(frequency_bands{i});
    for j = 1:length(frequency_bands)
        high_band = bandWidth(frequency_bands{j});
        if (i < j)            
            higher_freq_low = high_band(1); higher_freq_high = high_band(2);
            lower_freq_low = low_band(1); lower_freq_high = low_band(2);
            CFC_SI{i,j} = compute_CFC(subject_number,session_number,higher_freq_low,higher_freq_high,lower_freq_low,lower_freq_high);
        end
    end
end

% for i = 1:5
%     for j = 1:5
%         if (i ~= j)
%             compute_CFC(lows[i], highs[i], lows[j], highs[j]);
%         end
%     end
% end
