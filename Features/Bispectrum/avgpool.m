function outA = avgpool(A,downsample_factor)
% 4 pixels comprising non-overlapping 2-by-2 neighbourhoods  
im_nw=A(1:downsample_factor:end,1:downsample_factor:end);
im_sw=A(downsample_factor:downsample_factor:end,1:downsample_factor:end);
im_se=A(downsample_factor:downsample_factor:end,downsample_factor:downsample_factor:end);
im_ne=A(1:downsample_factor:end,downsample_factor:downsample_factor:end);

% Get average intensity:
% outA = max(cat(3,im_nw,im_sw,im_se,im_ne),[],3); % Change to this if want max-pooling
outA = nanmean(cat(3,im_nw,im_sw,im_se,im_ne),3); % Change this to max if want max-pooling