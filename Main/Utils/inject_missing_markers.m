function [EEG] = inject_missing_markers(EEG,eeg_srate,marker,time_pts,mri_TR)

%% initial view of number of markers, determination of latency units

%careful with the below code
if sum([EEG.event.type]==marker) == 0
    for i = 1:size([EEG.event],2)
        if EEG.event(i).type == 100008
            EEG.event(i).type = marker;
        end
    end
end

mymat = [EEG.event.type]==marker;
latency = [];
counter = 0;
for i = 1:size(mymat,2)
    if mymat(i) == 1
        counter = counter + 1;
        latency_end = EEG.event(i).latency;
        if counter == 1
            latency_start = latency_end;
        elseif counter > 1
            latency(end+1) = latency_end - latency_start;
            latency_start = latency_end;
        end
    end
end
num_markers = sum([EEG.event.type]==marker);

%logic start
if num_markers ~= time_pts
    % time_btwn_markers = mode(latency)/eeg_srate; %in seconds
    time_btwn_markers = mri_TR; %in seconds -- need to look into TR def'n
    newdata2=EEG.event;
    mymat = [EEG.event.type];
    latency = [];
    counter = 0;
    num_of_total_injections = 0;
    for i = 1:size(mymat,2)
        if mymat(i) == marker
            counter = counter + 1;
            latency_end = EEG.event(i).latency;
            if counter == 1
                latency_start = latency_end;
            elseif counter > 1
                latency(end+1) = latency_end - latency_start;
                if (latency_end - latency_start)/eeg_srate ~= time_btwn_markers
                    num_markers = round(((latency_end - latency_start)/eeg_srate)/time_btwn_markers);
                    curr_latency = EEG.event(i).latency/eeg_srate;
                    injection_progress = 0;
                    for j = 1:num_markers-1
                        injected_latency = (curr_latency - (time_btwn_markers*(num_markers-(injection_progress+1))))*eeg_srate;
                        s.type = marker;
                        s.latency = injected_latency;
                        s.urevent = 999;
                        newdata2 = [newdata2(:,1:(i+num_of_total_injections)-1+injection_progress),s,newdata2(:,(i+num_of_total_injections)+injection_progress:end)];
                        injection_progress = injection_progress +1;
                    end
                    num_of_total_injections = num_of_total_injections + num_markers-1;
                end
                latency_start = latency_end;
            end
            s.latency = EEG.event(i).latency;
        end
    end

    while sum([newdata2.type]==marker) ~= time_pts
        s.latency = s.latency + time_btwn_markers*eeg_srate;
        newdata2 = [newdata2(:,1:end),s];
    end

    EEG.event = newdata2;

    %re-number uevent param
    for i = 1:size([EEG.event],2)
        EEG.event(i).urevent = i;
    end
end
