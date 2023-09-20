% Use this function to find an EEG chunk in a larger Full EEG dataset
function start_idx = EEGfMRI_find_overlap(full_EEG,DATA_chunk,ch)
signalA = (full_EEG.data(ch,:)); signalB = DATA_chunk(ch,:);
start_idx = compute_lag(signalA,signalB);
figure; plot(signalB); hold on; plot(signalA(start_idx:start_idx+length(signalB)))