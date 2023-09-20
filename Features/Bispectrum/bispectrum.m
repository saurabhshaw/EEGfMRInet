function [Bspec_features_raw_cell, Bspec_features_real_cell, Bspec_features] = bispectrum(dataset, window_length, window_step)

start_idx = 1:window_step:size(dataset,2); 
end_idx = start_idx + window_length - 1;

while(end_idx(end) > size(dataset,2))
    start_idx=start_idx(1:end-1);
    end_idx=end_idx(1:end-1);
end 

Bspec_features_raw_cell = cell(1,length(start_idx));
Bspec_features_real_cell = cell(1,length(start_idx));

parfor i = 1:length(start_idx)
    [Bspec_curr, waxis] = bispecd(squeeze(dataset(:, start_idx(i):end_idx(i))));
    Bspec_features_raw_cell{i} = avgpool(Bspec_curr,downsample_factor);
    Bspec_features_real_cell{i} = reshape(real(Bspec_features_raw_cell{i}),[size(Bspec_features_raw_cell{i},1)*size(Bspec_features_raw_cell{i},2), 1]);
    % Bspec_features_imag_cell{i} = reshape(imag(Bspec_features_raw_cell{i}),[1,size(Bspec_features_raw_cell{i},1)*size(Bspec_features_raw_cell{i},2)]);
    % Bspec_features_cell{i} = [Bspec_features_real_cell{i}, Bspec_features_imag_cell{i}];
end
Bspec_features = cell2mat(Bspec_features_real_cell);

end



