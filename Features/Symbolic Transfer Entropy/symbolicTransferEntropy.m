function [STE, NSTE] = symbolicTransferEntropy(dataset, window_length, window_step)

start_idx = 1:window_step:size(dataset,2); 
end_idx = start_idx + window_length - 1;

while(end_idx(end) > size(dataset,2))
    start_idx=start_idx(1:end-1);
    end_idx=end_idx(1:end-1);
end 

STE = cell(1,length(start_idx));
NSTE = cell(1,length(start_idx));

for j = 1:length(start_idx)
   [STE{j}, NSTE{j}] = ste_eegapp(dataset, start_idx(j), end_idx(j));
end

end