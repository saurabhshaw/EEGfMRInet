function [arr_plotamp, arr_avgamp] = phase_amplitude_coupling_function_custom_bichannel(dataset,higher_freq_idx,lower_freq_idx,PAC_opt)
    %This function is called by the main function when PAC is selected
    %and the launch analysis button is pressed
    
try    
    
    channels_pac = 1:size(dataset{1},1); 
    numberOfBin = PAC_opt.numberOfBin; 
    window_length = floor(size(dataset{1},2)/PAC_opt.data_length_factor);

    total_pnts = size(dataset{1},2);    
    
    % Check to make sure we have reasonable segment numbers
    if(window_length <= total_pnts)
        segment_num = total_pnts/window_length;
    else
        segment_num = 1; 
    end
    
    % Here we calculate many points there are in a segment 
    if(segment_num ~= 1)
        segment_pnts = floor(total_pnts/segment_num);
    else
        segment_pnts = total_pnts - 1;
    end

    % Calculate the size of the bins
    segment_num = floor(segment_num);
    binsize = 2*pi/(numberOfBin);
    
    % Here we extract the phase and the amplitude from LFO and HFO
    phase = angle(dataset{lower_freq_idx}); % Take the angle of the Hilbert to get the phase of lower frequency
    amp = abs(dataset{higher_freq_idx}); % calculating the amplitude by taking absolute value of hilbert of the higher frequency
    % total_plot(segment_num,numberOfBin) = zeros();

    
    % Set up sortamp, plotamp, avgamp vector length segment_num
    % arr_sortamp = zeros(numberOfBin, 2, segment_num,length(channels_pac),length(channels_pac));
    arr_plotamp = zeros(numberOfBin,segment_num,length(channels_pac),length(channels_pac));
    arr_avgamp = zeros(segment_num,length(channels_pac),length(channels_pac));
    
    % Repeat the calculation below for each segment
    for segment = 1:segment_num
        % ch_sortamp = zeros([numberOfBin, 2, size(channels_pac,1), size(channels_pac,1)]);
        for n1 = 1:length(channels_pac)    % Find average over all channels
            
            % Put in another loop here to loop through other channels
            % SAURABH
            
            for n2 = 1:length(channels_pac)    % Find average over all channels
                
                ch_phase = phase(n1,:); % Rows are the low frequency phase
                ch_amp = amp(n2,:); % Columns are the high frequency amplitude
                
                %   Sort amplitudes according to phase.  Sortamp adds the amplitude in
                %   (:,1) and keeps track of the total numbers added in (:,2)
                sortamp = zeros([numberOfBin, 2]);
                sub = ((segment*segment_pnts)+1);
                if(segment==segment_num)
                    sub = total_pnts;
                end
                for i = (((segment-1)*segment_pnts)+1):sub
                    for j = 1:numberOfBin
                        if ch_phase(i) > (-pi+(j-1)*binsize) && ch_phase(i) < (-pi+(j*binsize))
                            sortamp(j,1) = sortamp(j,1) + ch_amp(i);
                            sortamp(j,2) = sortamp(j,2) + 1;
                            % ch_sortamp(j,1,n1,n2) = ch_sortamp(j,1,n1,n2) + ch_amp(i);
                            % ch_sortamp(j,2,n1,n2) = ch_sortamp(j,2,n1,n2) + 1;
                            break;
                        end
                    end
                end
                %end % Original channel end - SAURABH commented out
                
                %   Calculate average amplitude;
                avgsortamp = zeros(1,numberOfBin);
                for i = 1:numberOfBin
                    if sortamp(i,2) == 0
                        avgsortamp(i) = 0;
                    else
                        avgsortamp(i) = (sortamp(i,1)/sortamp(i,2));
                    end
                end
                
                avgamp = mean(avgsortamp);
                plotamp = zeros(1,numberOfBin);
                
                %For each bins set the value at that position
                for i = 1:numberOfBin
                    plotamp(i) = (avgsortamp(i)-avgamp)/avgamp + 1;
                end
                
                plotamp = plotamp - 1;            % Do this because median filter assumes 0 on each side
                plotamp = medfilt1(plotamp, 2);   % January 16, 2014
                plotamp = plotamp + 1;
                
                %         %Create the plot segment by segment
                %         for i = 1:numberOfBin
                %             total_plot(segment,i) = plotamp(1,i);
                %         end
                
                % arr_sortamp(:,:,segment,n1,n2) = squeeze(sortamp);
                arr_plotamp(:,segment,n1,n2) = squeeze(plotamp);
                arr_avgamp(segment,n1,n2) = squeeze(avgamp);
                
            end
        end        
    end
    
    arr_avgamp = squeeze(mean(arr_avgamp,1));
    
catch Exception
%     errors = 1;

    return
end

    
end