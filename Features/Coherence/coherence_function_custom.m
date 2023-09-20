function [avgCXY, all_coherence] = coherence_function_custom(EEGdata_filt,srate)
%Outputs an array of avgCXY & all_coherence where each cell contains
%info computed for the BP's selected. Will be in order listed prior.
%This function is called by the main function when the coherence is selected
%and the launch analysis button is pressed

try    
    from = 1:size(EEGdata_filt{1},1);
    to = 1:size(EEGdata_filt{1},1);
    
    % Calculating the different frequency bands:
    freq_number = length(EEGdata_filt);
    avgCXY = cell(1, freq_number);
    all_coherence = cell(1, freq_number);
    
    for sub = 1:freq_number % Iterate through the different frequency bands
        set(0,'DefaultFigureVisible','off');%disable output to screen
        
        data = EEGdata_filt{sub}';
        index_ch1 = 1;
        avgCXY{sub} = zeros(length(from),length(to));
        [CXY,~] = mscohere(1:size(data,1),1:size(data,1),[],[],[],srate);
        all_coherence{sub} = zeros(length(from), length(to), length(CXY'));      
        
        % Most efficient computation
        for ch1 = 1:length(from)
            d1 = repmat(data(:, from(ch1)),[1,length(from)]);
            [CXY, ~] = mscohere(data, d1, [], [], [], srate);
            all_coherence{sub}(ch1,:,:) = CXY';
            avgCXY{sub}(ch1,:) = mean(CXY,1);
        end

%         % Lesser efficiency:
%         for ch1 = 1:length(from);
%             for ch2 = 1:length(to);
%                 if ch1 < ch2
%                     d1 = data(:, from(ch1));
%                     d2 = data(:, to(ch2));
%                     [CXY, ~] = mscohere(d1, d2, [], [], [], srate);
%                     all_coherence{sub}(ch1,ch2,:) = CXY;
%                     avgCXY{sub}(ch1, ch2) = mean(CXY);
%                 end
%             end
%         end
        
    end
    
catch Exception
    warndlg('Coherence ran into some trouble, please click help->documentation for more information on Coherence.','Errors')
    disp(Exception.getReport());
    return
end

return

end