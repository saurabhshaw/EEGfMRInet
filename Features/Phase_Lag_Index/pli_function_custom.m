function [PLI, PLIcorr, PLI_z_score] = pli_function_custom(EEGdata_filt_hilbert, pli_prp)

try    
    % We load all variables needed for the analysis    
    pts_length = floor(size(EEGdata_filt_hilbert{1},2)/pli_prp.data_length_factor);  
    number_permutation = pli_prp.permutation;  
    p_value = pli_prp.p_value;

    % Calculating the different frequency bands:
    freq_number = length(EEGdata_filt_hilbert);
    
    % Calculate the maximum number of segment to calculate    
    maximum = floor(size(EEGdata_filt_hilbert{1},2) / pts_length); 
    
    PLI = cell(1, freq_number);
    PLIcorr = cell(1, freq_number);
    PLI_z_score = cell(1, freq_number);
     
    for sub = 1:freq_number % Iterate through the different frequency bands
        
        data = EEGdata_filt_hilbert{sub}';
        PLI_surr = zeros(maximum,number_permutation,size(EEGdata_filt_hilbert{1},1),size(EEGdata_filt_hilbert{1},1)); %initialize the matrix
        PLI{sub} = zeros(maximum,size(EEGdata_filt_hilbert{1},1), size(EEGdata_filt_hilbert{1},1)); % initialize matrix
        PLIcorr{sub} = zeros(maximum,size(EEGdata_filt_hilbert{1},1), size(EEGdata_filt_hilbert{1},1)); % initialize matrix
        
        for i = 1:maximum % Iterate through the different segments            
            
            data_eeg = data(1+(i-1)*pts_length:i*pts_length,:);            
            PLI{sub}(i,:,:) = w_PhaseLagIndex_custom(data_eeg); %calculate weighted pli

            %Calculating the surrogate
            for j = 1:number_permutation
                PLI_surr(i,j,:,:) = w_PhaseLagIndex_surrogate_custom(data_eeg);
            end
            
            %Here we compare the calculated PLI versus the surrogate
            %and test for significance
            for m = 1:size(PLI{sub},2)
                for n = 1:size(PLI{sub},3)
                    test = PLI_surr(i,:,m,n);
                    p = signrank(test, (PLI{sub}(i,m,n)));       
                    
                    if p < p_value
                        if PLI{sub}(i,m,n) - median(test) < 0 %Special case to make sure no PLI is below 0
                            PLIcorr{sub}(i,m,n) = 0;
                        else
                            PLIcorr{sub}(i,m,n) = PLI{sub}(i,m,n) - median(test);
                        end
                    else
                        PLIcorr{sub}(i,m,n) = 0;
                    end          
                end
            end
        end
        
        % Not more efficient:
%         % PLI_surr_cell = squeeze(mat2cell(PLI_surr,size(PLI_surr,1),ones(1,size(PLI_surr,2)),ones(1,size(PLI_surr,3))));
%         PLI_surr_cell = squeeze(num2cell(PLI_surr,2));
%         PLI_sub_cell = num2cell(squeeze(PLI{sub}));
%         p_test_cell = cellfun(@(x,y) signrank(x,y) ,PLI_surr_cell,PLI_sub_cell,'UniformOutput',0); 
%         p_test_cell_bin = num2cell(cell2mat(p_test_cell) < p_value);
%         median_test_cell_bin = cellfun(@(x,y) (y - median(x) >= 0),PLI_surr_cell,PLI_sub_cell,'UniformOutput',0);
%         PLIcorr{sub} = cell2mat(cellfun(@(x,y,m,n) (y - median(x))*m*n,PLI_surr_cell,PLI_sub_cell,p_test_cell_bin,median_test_cell_bin,'UniformOutput',0));
        
        % The z_score is the average of all segments 
        PLI_z_score{sub} = squeeze(mean(PLIcorr{sub},1));
%         PLI_z_score{sub} = zeros(size(EEGdata_filt_hilbert{1},1), size(EEGdata_filt_hilbert{1},1));
%         for a = 1:maximum
%             for i = 1:size(PLI{sub},2)
%                 for j = 1:size(PLI{sub},3)
%                     PLI_z_score{sub}(i,j) = PLI_z_score{sub}(i,j) + PLIcorr{sub}(a,i,j);
%                 end
%             end
%         end
%         
%         PLI_z_score{sub} = PLI_z_score{sub}/maximum;
        
        

    end
    
catch Exception

    return
end

return    
end

function [] = topofunction(varargin)
%TOPOFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    S = varargin{3};  % Get the structure.
    s = varargin{5};
    current_EEG=varargin{4};
    
F = get(S.fh,'currentpoint');  % The current point w.r.t the figure.
% Figure out of the current point is over the axes or not -> logicals.
tf1 = S.AXP(1) <= F(1) && F(1) <= S.AXP(1) + S.AXP(3);
tf2 = S.AXP(2) <= F(2) && F(2) <= S.AXP(2) + S.AXP(4);

if tf1 && tf2
    % Calculate the current point w.r.t. the axes.
    Cx =  ceil(S.XLM(1) + (F(1)-S.AXP(1)).*(S.DFX/S.AXP(3)));
    Cy =  ceil(S.YLM(1) + (F(2)-S.AXP(2)).*(S.DFY/S.AXP(4)));
    Cy = size(current_EEG,1) - Cy + 1;
end

orderType = evalin('base','orderType');
if(strcmp(orderType,'custom') == 1)
    order = evalin('base','newOrder');
    Cx = order(Cx,1);
    Cy = order(Cy,1);
end

for i = 1:size(current_EEG,1)
    
    if( i == Cx || i == Cy)
        %display(i);
    else
       current_EEG.chanlocs(i).labels = ' ';
    end
end

cla(s(2))
subplot(s(2))
topoplot([],current_EEG.chanlocs,'style','blank','electrodes','labelpoint','chaninfo',current_EEG.chaninfo);
title('Left click on the PLI matrix to see the channels in the topographic plot');
end
