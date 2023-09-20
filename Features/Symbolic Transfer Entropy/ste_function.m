 function [STE, NSTE] = ste_function(dataset,ste_prp,workingDirectory)
    %This function is called by the main function when the STE is selected
    %and the launch analysis button is pressed
      
try 

    full_bp = bandWidth('full');
    alpha_bp = bandWidth('alpha');
    beta_bp = bandWidth('beta');
    delta_bp = bandWidth('delta');
    theta_bp = bandWidth('theta');
    gamma_bp = bandWidth('gamma');
        
    if(size(dataset,1)>100)     %Determines whether the incoming data is fmri of EEG
        SF = 0.5;
    else
        SF=500;
    end 
    
    %Preparing the EEG data
    samp_freq = SF;
    data = double(dataset);
    data = data';
    %[m, num_comp] = size(data);
    
    %set up the variables needed for ste analysis
    winsize=(ste_prp.winsize); 
    TotalWin=floor(length(data)/winsize); % Total number of window
    NumWin= ste_prp.numberwin ;
    RanWin=randperm(TotalWin); % Randomize the order
    UsedWin=RanWin(1:NumWin); % Randomly pick-up the windows
    UsedWin=sort(UsedWin);
    
    dim= ste_prp.dim;
    tau=ste_prp.tau;
        
    full_ste = ste_prp.full;
    delta_ste = ste_prp.delta;
    theta_ste = ste_prp.theta;
    alpha_ste = ste_prp.alpha;
    beta_ste = ste_prp.beta;
    gamma_ste = ste_prp.gamma;
    
    from = ste_prp.fromchan;
    to = ste_prp.tochan;

    %Calculating the number of plots needed
    plot_number = full_ste + delta_ste + theta_ste + alpha_ste + beta_ste + gamma_ste;    
    STE = cell(1, plot_number);
    NSTE = cell(1, plot_number);
    
    %Calculation of the STE will be done one bandpass after the other
    for sub=1:plot_number

        % Here we choose the low pass and high pass values for this iteration        
        if full_ste == 1
            lp = full_bp(1,1); 
            hp = full_bp(1,2);
        elseif delta_ste == 1
            lp = delta_bp(1,1); 
            hp = delta_bp(1,2);
        elseif theta_ste == 1
            lp = theta_bp(1,1); 
            hp = theta_bp(1,2);    
        elseif alpha_ste == 1
            lp = alpha_bp(1,1); 
            hp = alpha_bp(1,2);    
        elseif beta_ste == 1
            lp = beta_bp(1,1); 
            hp = beta_bp(1,2);    
        elseif gamma_ste == 1
            lp = gamma_bp(1,1); 
            hp = gamma_bp(1,2);    
        end  
    
        %Calculate STE for every source channels to every sink channels
        %And for every sink channels to every source channels
        
        STE{sub} = NaN(NumWin,size(dataset,1),size(dataset,1));
        NSTE{sub} = NaN(NumWin,size(dataset,1),size(dataset,1));
        
        for ch1=1:size(dataset,1)
            %If ch1 is in from then do this loop
            if(any(ch1==from))
            
            for ch2=1:size(dataset,1)
                %if ch2 is in to then do this loop
                if(any(ch2==to))
                STE{sub} = NaN(15,2);
                NSTE{sub} = NaN(15,2);
                              
                for m=1:NumWin
                    win=UsedWin(m);
                    ini_point=(win-1)*winsize+1;
                    final_point=ini_point+winsize-1;
                    
                    x=data(ini_point:final_point,ch1);
                    y=data(ini_point:final_point,ch2);
                    
                    fdata1=bpfilter(lp,hp,samp_freq,x);
                    fdata2=bpfilter(lp,hp,samp_freq,y);
                    
                    delta=f_predictiontime(fdata1,fdata2,100); %Maybe something here

                    for L=1:14
                        [STE{sub}(L,1:2), NSTE{sub}(L,1:2)] = f_nste([fdata1 fdata2], dim, tau(L), delta);
                    end
                                       
                    [mxNSTE, ~]=max(NSTE{sub}); %mxNSTE and mxNTau
                    [mxSTE, ~]=max(STE{sub}); 
                    
                    STE{sub}(m,ch2,ch1)=mxSTE(1);    % Sink to Source
                    NSTE{sub}(m,ch2,ch1)=mxNSTE(1);
                    
                    STE{sub}(m,ch1,ch2)=mxSTE(2);    % Source to Sink
                    NSTE{sub}(m,ch1,ch2)=mxNSTE(2);
                           
                    %Update the waitbar
                end
                end 
            
            end
            end
        end
        
        %Here we turn off those the bandpass we already did
        if full_ste == 1
            full_ste = 0;
        elseif delta_ste == 1
            delta_ste = 0;
        elseif theta_ste == 1
            theta_ste = 0;
        elseif alpha_ste == 1
            alpha_ste = 0;
        elseif beta_ste == 1
            beta_ste = 0;
        elseif gamma_ste == 1
            gamma_ste = 0;
        end

end
    
catch Exception
    warndlg('Symbolic Transfer Entropy ran into some trouble, please click help->documentation for more information on Symbolic Transfer Entropy.','Errors')
    disp(Exception.getReport());
    return
end

return        
 end