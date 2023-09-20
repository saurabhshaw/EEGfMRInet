function [Xcsp,F,cspftrs,Psel] = FBCSPfeatures(X,Y,indtrain,opt)
% -----------------------------------------------------------------------
% Description: 
% This function computes the csp feature vectors for each sub-band of 
% of filtered EEG and provides the FBCSP filters and features. 
% -----------------------------------------------------------------------
% Dependancies: 
% csp_base
% csp_base_cell
% -----------------------------------------------------------------------
% Inputs: 
% X: nChans x nSamps x nTrials EEG data (can be a 1 x nTrials cell array)
% Y: 1 x nTrials vector of class labels
% indtrain: a vector containing the indices for training trials
% opt: a structure specifying csp options and bandpass filter options
%   opt.features.cspcomps: number of CSP components from which to compute
%   features (uses top 2 and bottom 2 cspcomps components, so cspcomps must
%   be at most nChans/2)
%   opt.features.cspband: the total frequency band over which to perform FBCSP
%   opt.features.cspwidth: the width of each bandpass filter in the filter bank
%   opt.features.cspstep: shift in Hz between adjacent bandpass filters
%   opt.mode: method for computing base csp (see csp_base.m)
%   opt.filter.type: type of bandpass filter ('butter','cheby2','fir')
%   opt.filter.order: order of the bandpass filter
%   opt.filter.sr: sampling rate of the input EEG signal
% 
% -----------------------------------------------------------------------
% Outputs: 
% Xcsp: The input EEG signals in X transformed by each CSP filter
% F: CSP features formatted in a standard nFeat x nData feature matrix
% cspftrs: the same CSP features organized by each filter
% Psel: The CSP filters (weight matrices) for each bandpass filter
% 
% -----------------------------------------------------------------------
% References:
% [1]   Ang, K.K., Chin, Z. Y., Zhang, H., and Guan, C. (2008). 
%       Filter Bank COmmon Spatial Patterns (FBCSP). International Joint
%       Conference on Neural Networks Pp. 2390-2397.
% -----------------------------------------------------------------------
% Author: Kiret Dhindsa
%------------------------------------------------------------------------

if ismatrix(X)&&~iscell(X)
    % Indexing
    [nChans,nSamps,nTrials] = size(X);
    nBands = ((opt.features.cspband(2)-(opt.features.cspwidth-opt.features.cspstep))...
        - opt.features.cspband(1))/opt.features.cspstep;
    
    % Init outputs
    P = zeros(2*opt.features.cspcomps,nChans,nBands);
    cspftrs = zeros(2*opt.features.cspcomps,nBands,nTrials);
    Xcsp = zeros(2*opt.features.cspcomps,nSamps,nTrials,nBands);
    
    % Initial frequency band
    low = opt.features.cspband(1);
    high = low + opt.features.cspwidth;
    
    % Normalize all trials
    Xnorm = zeros(size(X));
    parfor t = 1: nTrials
        Xnorm(:,:,t) = bsxfun(@minus,X(:,:,t),mean(X(:,:,t),2));
    end
    
    for band = 1:nBands
        % Bandpass filter normalized trials
        if strcmp(opt.filter.type,'butter')
            [b,a] = butter(opt.filter.order,[low,high]/(opt.filter.sr/2));
        elseif strcmp(opt.filter.type,'cheby2')
            [b,a] = cheby2(opt.filter.order,opt.filter.atten,[low,high]/(opt.filter.sr/2));
        elseif strcmp(opt.filter.type,'fir')
            b = fir1(opt.filter.order,[low,high]/(opt.filter.sr/2));
            a = 1;
        end
        
        Xfilt = zeros(size(Xnorm));
        parfor tr = 1:nTrials
            Xfilt(:,:,tr) = filter(b,a,Xnorm(:,:,tr)')';
        end
        
        % CSP projection matrix
        P(:,:,band) = csp_base(Xfilt(:,:,indtrain),Y(indtrain),opt);
        
        % Project Signals
        Xproj = zeros(size(P,1),nSamps,nTrials);
        parfor tr = 1:nTrials
            Xproj(:,:,tr) = P(:,:,band)*Xfilt(:,:,tr);
        end
        
        % Compute Variances
        Xvars = zeros(size(P,1),nTrials);
        parfor tr = 1: nTrials
            Xvars(:,tr) = var(Xproj(:,:,tr),[],2);
        end
        
        % Compute CSP features
        parfor tr = 1: nTrials
            cspftrs(:,band,tr) = log10(Xvars(:,tr)./sum(Xvars(:,tr)));
        end
        
        % Update frequency band and Output args
        low = low + opt.features.cspstep;
        high = high + opt.features.cspstep;
        Xcsp(:,:,:,band) = Xproj;
    end
    
    % Output
    Psel = P;
    F = reshape(cspftrs,2*opt.features.cspcomps*nBands,nTrials);
    F_isInf = sum(arrayfun(@isinf,F),2);
    F_isNaN = sum(arrayfun(@isnan,F),2);
    F = F(~(F_isInf | F_isNaN),:);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif iscell(X)
    % Indexing
    nTrials = length(X);
    nChans = size(X{1},2);
    nBands = ((opt.features.cspband(2)-(opt.features.cspwidth-opt.features.cspstep))...
        - opt.features.cspband(1))/opt.features.cspstep;
    
    % Init outputs
    P = zeros(2*opt.features.cspcomps,nChans,nBands);
    cspftrs = zeros(2*opt.features.cspcomps,nBands,nTrials);
    Xcsp = cell(1,nTrials);
    
    % Initial frequency band
    low = opt.features.cspband(1);
    high = low + opt.features.cspwidth;
    
    % Normalize all trials
    Xnorm = cell(1,nTrials);
    parfor t = 1: nTrials
        Xnorm{t} = bsxfun(@minus,X{t},mean(X{t}));
    end
    
    for band = 1:nBands
        % Bandpass filter normalized trials
        if strcmp(opt.filter.type,'butter')
            [b,a] = butter(opt.filter.order,[low,high]/(opt.filter.sr/2));
        elseif strcmp(opt.filter.type,'cheby2')
            [b,a] = cheby2(opt.filter.order,opt.filter.atten,[low,high]/(opt.filter.sr/2));
        elseif strcmp(opt.filter.type,'fir')
            b = fir1(opt.filter.order,[low,high]/(opt.filter.sr/2));
            a = 1;
        end
        
        Xfilt = cell(1,nTrials);
        parfor tr = 1:nTrials
            Xfilt{tr} = filter(b,a,Xnorm{tr});
        end
        
        % CSP projection matrix
        P(:,:,band) = csp_base_cell(Xfilt(indtrain),Y(indtrain),opt);
        
        % Project Signals
        Xproj = cell(1,nTrials);
        parfor tr = 1:nTrials
            Xproj{tr} = P(:,:,band)*Xfilt{tr}';
        end
        
        % Compute Variances
        Xvars = zeros(size(P,1),nTrials);
        parfor tr = 1: nTrials
            Xvars(:,tr) = var(Xproj{tr},[],2);
        end
        
        % Compute CSP features
        parfor tr = 1: nTrials
            cspftrs(:,band,tr) = log10(Xvars(:,tr)./sum(Xvars(:,tr)));
        end
        
        % Update frequency band and Output args
        low = low + opt.features.cspstep;
        high = high + opt.features.cspstep;
        Xcsp(:,:,:,band) = Xproj;
    end
    
    % Output
    Psel = P;
    F = reshape(cspftrs,2*opt.features.cspcomps*nBands,nTrials);    
    F_isInf = sum(arrayfun(@isinf,F),2);
    F_isNaN = sum(arrayfun(@isnan,F),2);
    F = F(~(F_isInf | F_isNaN),:);
end

