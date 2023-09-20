function [SI, SI_mag, SI_theta] = compute_CFCSI(dataset,higher_freq_idx,lower_freq_idx)

num_chan = size(dataset{1},1);

hilbert_higherFiltdata = dataset{higher_freq_idx}';
hilbert_lowerFiltdata = dataset{lower_freq_idx}';

% Calculate Hilbert transform (Real and Imaginary components)
power_higherFiltdata = abs(hilbert_higherFiltdata);
hilbertpower_higherFiltdata = hilbert(power_higherFiltdata);

% Normalize and demean Hilbert transform
%window_hilbert_higherFiltdata(chan,j,:) = window_hilbert_higherFiltdata(chan,j,:)./norm(squeeze(window_hilbert_higherFiltdata(chan,j,:))); window_hilbert_higherFiltdata(chan,j,:) = window_hilbert_higherFiltdata(chan,j,:) - mean(window_hilbert_higherFiltdata(chan,j,:));
%window_hilbert_lowerFiltdata(chan,j,:) = window_hilbert_lowerFiltdata(chan,j,:)./norm(squeeze(window_hilbert_lowerFiltdata(chan,j,:))); window_hilbert_lowerFiltdata(chan,j,:) = window_hilbert_lowerFiltdata(chan,j,:) - mean(window_hilbert_lowerFiltdata(chan,j,:));

% Calculate Phase from Real and Imaginary components:
theta_higherFiltdata = angle(hilbertpower_higherFiltdata);
theta_lowerFiltdata = angle(hilbert_lowerFiltdata);
% theta_higherFiltdata = atan(imag(hilbertpower_higherFiltdata)./real(hilbertpower_higherFiltdata));
% theta_lowerFiltdata = atan(imag(hilbert_lowerFiltdata)./real(hilbert_lowerFiltdata));

% SI = (1/size(dataset{1},2)).*sum(exp(1i.*(theta_lowerFiltdata - theta_higherFiltdata)));

% Calculate synchronization index (SI):
SI = zeros(num_chan,num_chan);
SI_mag = zeros(num_chan,num_chan);
SI_theta = zeros(num_chan,num_chan);

for ch1 = 1:num_chan
    d1 = repmat(theta_lowerFiltdata(:,ch1),[1,num_chan]);
    SI(ch1,:) = (1/size(dataset{1},2)).*sum(exp(1i.*(d1 - theta_higherFiltdata)));
    SI_mag(ch1,:) = abs(SI(ch1,:));
    SI_theta(ch1,:) = angle(SI(ch1,:));
end

end
