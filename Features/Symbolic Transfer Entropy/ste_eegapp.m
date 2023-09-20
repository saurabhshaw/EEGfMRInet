function [STE, NSTE] = ste_eegapp(dataset, start, stop)
%   Input:
%   winsize = size of the windows to calculate ste. 
%   NumWin = number of windows to calculate ste.

%   STE always computes to zero in this example, NSTE does not however.
%   Code gets error when too many channels selected, too many bp's
%   calculated, hence it may be due to computational limitations.

    winsize = floor((stop-start+1)/3);  
    numberwin = 3;        
        
    fromchan = 1:size(dataset,1);
    tochan = 1:size(dataset,1);

    dim = 2;
    tau = 1:2:30;
    
    full = 1;
    delta = 1;
    theta = 1;
    alpha = 1;
    beta = 1;
    gamma = 1; 
    
    ste_prp = struct('winsize',winsize,'numberwin',numberwin,'fromchan',fromchan,...
    'tochan',tochan,'dim',dim,'tau',tau,'print',0,'save',0,...
    'full',full,'delta',delta,'theta',theta,'alpha',alpha,...
    'beta',beta,'gamma',gamma);
    
    dataset=dataset(:,start:stop);

    [STE, NSTE] = ste_function(dataset, ste_prp, pwd);
   
end