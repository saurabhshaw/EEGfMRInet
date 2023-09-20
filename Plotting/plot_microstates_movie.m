%% Plotting Movies:
% peak_num = 14; num_peaks_traverse = 4;
% load('Movie_workspace.mat');
Fs = 500;
start_idx = gfp_peak_loc(peak_num); end_idx = gfp_peak_loc(peak_num+num_peaks_traverse);

% [Movie,Colormap] = eegmovie(data_input_active(start_idx:end_idx,:)',500,active_chan_locs,'Topology between the two GFP peaks',0,0);
 [Movie,Colormap] = eegmovie(data_trunc(start_idx:end_idx,:)',Fs,active_chan_locs,'Topology between the two GFP peaks',0,0);
%[Movie,Colormap] = plot_microstates_movie_subplots(data_trunc(start_idx:end_idx,:)',500,active_chan_locs,'Topology between the two GFP peaks',0,0);

%% Compute Entropy:
half_window = 30;
E = [];
for jj = start_idx:end_idx
    E = [E wentropy(data_trunc(start_idx - half_window:start_idx + half_window,:),'shannon')];
end

%% Plotting Graphs:
% start_idx = gfp_peak_loc(peak_num); end_idx = gfp_peak_loc(peak_num+1);
% len_idx = start_idx:end_idx;
% figure; stem(1:len_idx,dist(1,start_idx:end_idx));

curr_time = double((start_idx:end_idx)./Fs);
figure; hax = axes; stem(curr_time,dist(1,start_idx:end_idx))
hold on; stem(curr_time,dist(2,start_idx:end_idx))
hold on; stem(curr_time,dist(3,start_idx:end_idx))
hold on; stem(curr_time,dist(4,start_idx:end_idx))
hold on; plot(curr_time,1000*gfp(start_idx:end_idx));
% hold on; plot(curr_time,1000*E);
% hold on; plot(curr_time(curr_peak_locs),1000*(gfp(curr_peak_locs)+ 1),'vk');

curr_peak_locs = [gfp_peak_loc(peak_num)];
for i = 1:num_peaks_traverse
    curr_peak_locs = [curr_peak_locs gfp_peak_loc(peak_num+i)];
end
curr_peak_locs = curr_peak_locs - start_idx + 1;

for j = 1:length(curr_peak_locs)
    line([curr_time(curr_peak_locs(j)) curr_time(curr_peak_locs(j))],get(hax,'YLim'),'Color',[1 0 0]);
end

%% Plot Velocity:

curr_time = (start_idx:end_idx)./Fs;
figure; hax = axes; stem(curr_time,vel(1,start_idx:end_idx))
hold on; stem(curr_time,vel(2,start_idx:end_idx))
hold on; stem(curr_time,vel(3,start_idx:end_idx))
hold on; stem(curr_time,vel(4,start_idx:end_idx))
hold on; plot(curr_time,100*gfp(start_idx:end_idx));
% hold on; plot(curr_time,0.0001*E);
% hold on; plot(curr_time(curr_peak_locs),1000*(gfp(curr_peak_locs)+ 1),'vk');

curr_peak_locs = [gfp_peak_loc(peak_num)];
for i = 1:num_peaks_traverse
    curr_peak_locs = [curr_peak_locs gfp_peak_loc(peak_num+i)];
end
curr_peak_locs = curr_peak_locs - start_idx + 1;

for j = 1:length(curr_peak_locs)
    line([curr_time(curr_peak_locs(j)) curr_time(curr_peak_locs(j))],get(hax,'YLim'),'Color',[1 0 0]);
end

%% Plot Acceleration:

curr_time = (start_idx:end_idx)./Fs;
figure; hax = axes; stem(curr_time,acc(1,start_idx:end_idx))
hold on; stem(curr_time,acc(2,start_idx:end_idx))
hold on; stem(curr_time,acc(3,start_idx:end_idx))
hold on; stem(curr_time,acc(4,start_idx:end_idx))
hold on; plot(curr_time,100*gfp(start_idx:end_idx));
% hold on; plot(curr_time,100*E);
% hold on; plot(curr_time(curr_peak_locs),1000*(gfp(curr_peak_locs)+ 1),'vk');

curr_peak_locs = [gfp_peak_loc(peak_num)];
for i = 1:num_peaks_traverse
    curr_peak_locs = [curr_peak_locs gfp_peak_loc(peak_num+i)];
end
curr_peak_locs = curr_peak_locs - start_idx + 1;

for j = 1:length(curr_peak_locs)
    line([curr_time(curr_peak_locs(j)) curr_time(curr_peak_locs(j))],get(hax,'YLim'),'Color',[1 0 0]);
end