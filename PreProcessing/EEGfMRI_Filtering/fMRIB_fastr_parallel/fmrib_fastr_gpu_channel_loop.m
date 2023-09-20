% fmrib_fastr() -  Remove FMRI gradient artifacts from EEG data using
%   FMRI artifact Slice Template Removal (FASTR) [Niazy06] and uses optimal
%   basis sets (OBS) to reduce the residuals.
%
%   This subtraction algorithm is based on the principles outlined by
%   [Niazy06] with few improvements and modifications. This program
%   constructs a unique artifact template for each slice then subtracts the
%   artifact.  Residual artifacts are removed by constructing a matrix of
%   the residuals, doing a PCA then fitting the first 4 PCs (Optimal basis
%   set - OBS)to the residuals in each slice.  This procedure should not be
%   applied for non-EEG channels (e.g. ECG) as it can remove some high frequency
%   details from a signal (e.g. QRS complex). Adaptive noise cancellation
%   (ANC) [Allen00] is then used.
%
%
%   USAGE:
%   EEG=fmrib_fastr(EEG,lpf,L,window,Trigs,strig,anc_chk,tc_chk,Volumes,Slices,varargin);
%   EEG:  EEGLAB data structure
%   lpf:  low-pass filter cutoff
%   L: Interpolation folds
%   window: length of averaging window in number of artifacts
%   Trigs: An array of slice triggers locations.
%   strig: 1 for slice triggers, 0 for volume / section triggers.
%   anc_chk: 1 to do Adaptive noise cancellation
%			 0 to not.
%   tc_chk:  1 to correct for missing triggers, 0 for not
%   Volumes: FMRI volumes for use in trigger correction
%   Slices:  FMRI Slices / Volume for use in trigger correction
%   varargin{1}: relative position of slice trigger from beginning of
%       slice acquisition: 0 for exact start -> 1 for exact end
%       default=0.03;
%   varargin{2}: Channels not to perform OBS  on.
%   varargin{3}: Numer of PCs to use in OBS. use 0 to skip this step.
%                'auto' or empty for auto order selection.
%
% [Niazy06] R.K. Niazy, C.F. Beckmann, G.D. Iannetti, J.M. Brady, and
%  S.M. Smith (2005) Removal of FMRI environment artifacts from EEG data
%  using optimal basis sets. NeuroImage 28 (3), pages 720-737.
%
%
% [Allen00] A Method for Removing Imaging Artifact from Continuous EEG
%   Recording during Functional MRI, P.J. Allen, O. Josephs, R. Turner.
%   NeuroImage 12, 230-239 (2000).
%
%
%
%
%   Author:  Rami K. Niazy, FMRIB Centre, Univ. of Oxford.
%
%   Copyright (c) 2006 University of Oxford

% Copyright (C) 2006 University of Oxford
% Author:   Rami K. Niazy, FMRIB Centre
%           rami@fmrib.ox.ac.uk
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


% 31 MAR 2006
% PCA of residual now done on HPF data at 70Hz.  This resolved overfitting
% issues, which sometimes caused removal of data.

% 10 MAR 2006
% Fixed potentially sereous bug related to command line input order of varargin

% 13 JAN 2006
% ECG Channel Artifact now not least-square fitted

% 13 JAN 2006
% Fixed Section Marker Limit for s=sections case

% 16 SEP 2005
% Fixed bug when processing non-eeg channels

% 15 SEP 2005
% Added code to lengthen the section to insure enough triggers are
% contained.

% 13 SEP 2005
% Edited use of 'find' to be backward compatible with MATLAB 6

% 02 SEP 2005
% Fixed typo bug when interpolation = 1;
% Fixed problem when using '/' with single precision numbers.

% 01 SEP 2005
% Commented out test code causing problems when not in 'auto' mode.

% 09 AUG 2005
% Automatic order selection of residual PCs added.

% 05 AUG 2005
% search window for max correlation now determined by original fs.
% Fixed Misc bugs for volume/section triggers
% ANC made optional.

% 12 MAR 2005
% Program made to work for interleaved recording / Volume triggers
% Fixed various bugs

% 23 JAN 2005
% Fixed rounding of Filter Order

% 14 JAN 2005
% Updated Help

% 23 DEC 2004
% Added use of binary
% adaptive noise cancellation
% binary corr (pcorr2)
% Updated (c) info

% 21 DEC 2004
% Fixed bug for
% sections==1
% made ANC called from file

% 17 DEC 2004
% Use CORRCOEF instead of
% CORR2 to eliminate need for
% image porc toolbox

% 14 DEC 2004
% Major Updates:
%   Added RAPCO.
%   Input for trig events replaced with an array of Trigs instead
%   to simplify scripting.
%   Other misc bugs fixed.

% 06 OCT 2004
% Edited for possibility of L=1
% i.e. no interpolation / decimation



function EEG = fmrib_fastr_gpu_channel_loop(EEGData,lpf,L,Window,Trigs,strig,anc_chk,tc_chk,Volumes,Slices,varargin)

pre_frac = 0.03;
npc = 'auto';
exc = [];
hpf = 70;
fs = fsrate;
SecT = 60;
if isempty(lpf)
    lpf = 0;
end

if anc_chk==1 && lpf==0
    lpf = 70;
end

if rem(Window,2) ~= 0
    Window = Window+1;
end

if strig==1
    hwinl = Window;
else
    hwinl = Window/2;
end

scount = 0;
[m n] = size(EEGData);
minorder = 15;
minfac = 3;
STARTFLAG = 0;
LASTFLAG = 0;
TH_SLOPE = 2;
TH_CUMVAR = 80;
TH_VAREXP = 5;

exc = unique([exc find(std(EEG.data')==0)]);

%% Construct Artifacts and Subtract
% This is where the code is parallelized
% ---------------------------------

n = length(EEGData);

try
    parfor c = 1:m        
        % Progress bar Init
%         if c==1
%             barth=5;
%             barth_step=barth;
%             Flag25=0;
%             Flag50=0;
%             Flag75=0;
%             fprintf('\nStage 2 of 2: Artifact Subtraction\n');
%         end

        % Initialize variables required by the parfor loop:
        
        if strig==1
            slice_art1=zeros(hwinl+1,art_length);
            slice_art2=zeros(hwinl+1,art_length);
        else
            slice_art1=zeros(2*hwinl+1,art_length);
        end
        avg_art=zeros(1,art_length);
        Noise=zeros(1,n);
        pcamat=zeros(floor(max(markerl)/2),pre_peak+max_postpeak+1);
        parsecmarker2 = secmarker2;
        STARTFLAG = 0;
        LASTFLAG = 0;
        
        % Zero-mean the EEGData for the current channel:
        tmpdata = EEGData{c} - mean(EEGData{c});
        
        cleanEEG = EEGData{c};
        
        % Process in sections of SecT seconds for memory concerns
        for sec=1:sections
            
            if L > 1
                if sec==1 & sections > 1
                    Idata = interp(tmpdata(d1:(d1-1)+secl(sec)+pad_sec),L,4,1);
                    Iorig = interp(EEGData{c}(d1:(d1-1)+secl(sec)+pad_sec),L,4,1);
                elseif sec==1 & sections==1
                    Idata = interp(tmpdata(d1:d2),L,4,1);
                    Iorig = interp(EEGData{c}(d1:d2),L,4,1);
                elseif sec==sections
                    Idata = interp(tmpdata((d1-1)+...
                        sum(secl(1:sec-1))+1-pad_sec:d2),L,4,1);
                    Iorig = interp(EEGData{c}((d1-1)+...
                        sum(secl(1:sec-1))+1-pad_sec:d2),L,4,1);
                else
                    Idata = interp(tmpdata((d1-1)+sum(secl(1:sec-1))+...
                        1-pad_sec:(d1-1)+sum(secl(1:sec))+pad_sec),L,4,1);
                    Iorig = interp(EEGData{c}((d1-1)+sum(secl(1:sec-1))+...
                        1-pad_sec:(d1-1)+sum(secl(1:sec))+pad_sec),L,4,1);
                end
            else
                if sec==1 & sections > 1
                    Idata = tmpdata(d1:(d1-1)+secl(sec)+pad_sec);
                    Iorig = EEGData{c}(d1:(d1-1)+secl(sec)+pad_sec);
                elseif sec==1 & sections==1
                    Idata = tmpdata(d1:d2);
                    Iorig = EEGData{c}(d1:d2);
                elseif sec==sections
                    Idata = tmpdata((d1-1)+sum(secl(1:sec-1))+1-pad_sec:d2);
                    Iorig = EEGData{c}((d1-1)+sum(secl(1:sec-1))+1-pad_sec:d2);
                else
                    Idata = tmpdata((d1-1)+sum(secl(1:sec-1))+1-pad_sec:...
                        (d1-1)+sum(secl(1:sec))+pad_sec);
                    Iorig = EEGData{c}((d1-1)+sum(secl(1:sec-1))+1-pad_sec:...
                        (d1-1)+sum(secl(1:sec))+pad_sec);
                end
            end
            
            INoise=zeros(1,length(Idata));
            
            %Average Artifacts & Subtract
            
            if sections==1
                starts=1;
                lasts=markerl(sec);
            elseif sec==1
                starts=1;
                lasts=markerl(sec)-2;
            elseif sec==sections
                starts=sum(markerl(1:sec-1))+2;
                lasts=sum(markerl(1:sec));
            else
                starts=sum(markerl(1:sec-1))+2;
                lasts=sum(markerl(1:sec))-2;
            end
            
            for s=starts:lasts
                curr_secmarker2 = parsecmarker2(s);
                
                
                if strig==1 % Slice Triggers
                    
                    if s==starts
                        art=1;
                        ssc=1;
                        for ss=starts+1:2:starts+2*hwinl+1
                            slice_art1(ssc,:)=...
                                Idata(secmarker(ss)-pre_peak:...
                                secmarker(ss)+post_peak);
                            ssc=ssc+1;
                        end
                        avg_art=mean(slice_art1,1);
                    elseif s==starts+1
                        ssc=1;
                        for ss=starts+2:2:starts+2*hwinl+2
                            slice_art2(ssc,:)=...
                                Idata(secmarker(ss)-pre_peak:...
                                secmarker(ss)+post_peak);
                            ssc=ssc+1;
                        end
                        avg_art=mean(slice_art2,1);
                    elseif ((s>(starts+hwinl+2)) & (s<=(lasts-(hwinl+2))))
                        ss=s+hwinl;
                        switch art
                            case 1
                                slice_art1=[slice_art1(2:end,:);...
                                    Idata(secmarker(ss)-pre_peak:...
                                    secmarker(ss)+post_peak)];
                                avg_art=mean(slice_art1,1);
                                art=2;
                            case 2
                                slice_art2=[slice_art2(2:end,:);...
                                    Idata(secmarker(ss)-pre_peak:...
                                    secmarker(ss)+post_peak)];
                                avg_art=mean(slice_art2,1);
                                art=1;
                        end
                    end
                    
                elseif strig==0  % Volume/Section Triggers
                    
                    if s==starts
                        art=1;
                        ssc=1;
                        for ss=starts+1:starts+2*hwinl+1
                            slice_art1(ssc,:)=...
                                Idata(secmarker(ss)-pre_peak:...
                                secmarker(ss)+post_peak);
                            ssc=ssc+1;
                        end
                        avg_art=mean(slice_art1,1);
                    elseif ((s>(starts+hwinl+2)) & (s<=(lasts-(hwinl+2))))
                        ss=s+hwinl;
                        slice_art1=[slice_art1(2:end,:);...
                            Idata(secmarker(ss)-pre_peak:...
                            secmarker(ss)+post_peak)];
                        avg_art=mean(slice_art1,1);
                    end
                end
                
                % For first channel, find shift in artifact position to minimise
                % sum of squared error between data and artifact template
                % - Assume same shift applies for all channels-
                % Also calculate Scale factor 'Alpha' to minimize sum of
                % squared error
                % ppn=1;
                if s==1
                    try
                        if c==1
                            B_idx = secmarker(s)-searchw:secmarker(s)+searchw;
                            C = zeros(1,length(B_idx));
                            for B=secmarker(s)-searchw:secmarker(s)+searchw
                                ppn = B - (secmarker(s)-searchw) + 1;
                                C(ppn)=prcorr2(Idata(B-pre_peak:B+post_peak),...
                                    avg_art);
                            end
                            parC = C;
                            [CV,CP]=max(parC);
                            Beta=CP-(searchw+1);
                            curr_secmarker2 = curr_secmarker2+Beta;
                        end
                        if isempty(intersect(exc,c))
                            Alpha=sum(Idata(curr_secmarker2-pre_peak:curr_secmarker2+...
                                post_peak).*avg_art)/sum(avg_art.*avg_art);
                        else
                            Alpha=1;
                        end
                        
                        INoise(curr_secmarker2-pre_peak:curr_secmarker2+...
                            post_peak)=Alpha*avg_art;
                        
                    catch
                        if c==1
                            if sec==1
                                warning...
                                    ('Not enough data to remove first artifact segment');
                                STARTFLAG=1;
                            end
                        end
                    end
                elseif sec==sections & s==lasts
                    try
                        if c==1
                            B_idx = secmarker(s)-searchw:secmarker(s)+searchw;
                            C = zeros(1,length(B_idx));
                            for B=secmarker(s)-searchw:secmarker(s)+searchw
                                ppn = B - (secmarker(s)-searchw) + 1;
                                C(ppn)=prcorr2(Idata(B-pre_peak:B+post_peak),...
                                    avg_art);
                            end
                            parC = C;
                            [CV,CP]=max(parC);
                            Beta=CP-(searchw+1);
                            curr_secmarker2=curr_secmarker2+Beta;
                        end
                        if isempty(intersect(exc,c))
                            Alpha=...
                                sum(Idata(curr_secmarker2-pre_peak:...
                                curr_secmarker2+post_peak).*avg_art)/...
                                sum(avg_art.*avg_art);
                        else
                            Alpha=1;
                        end
                        
                        if curr_secmarker2+post_peak <= length(Iorig) %fix so that when it goes beyond the end of the data, does not later on cause crash
                            INoise(curr_secmarker2-pre_peak:curr_secmarker2+post_peak)...
                                =Alpha*avg_art;
                        end;
                    catch
                        if c==1
                            warning('Not enough data to remove last artifact segment');
                            LASTFLAG=1;
                        end
                    end
                else
                    if c==1
                        B_idx = secmarker(s)-searchw:secmarker(s)+searchw;
                        C = zeros(1,length(B_idx));
                        for B=secmarker(s)-searchw:secmarker(s)+searchw
                            ppn = B - (secmarker(s)-searchw) + 1;
                            C(ppn)=prcorr2(Idata(B-pre_peak:B+post_peak),...
                                avg_art);
                        end
                        parC = C;
                        [CV,CP]=max(parC);
                        Beta=CP-(searchw+1);
                        curr_secmarker2=curr_secmarker2+Beta;
                    end
                    if isempty(intersect(exc,c))
                        Alpha=sum(Idata(curr_secmarker2-pre_peak:...
                            curr_secmarker2+post_peak).*avg_art)/...
                            sum(avg_art.*avg_art);
                    else
                        Alpha=1;
                    end
                    
                    INoise(curr_secmarker2-pre_peak:curr_secmarker2+post_peak)=...
                        Alpha*avg_art;
                end
                
                %parsecmarker2(s) = curr_secmarker2;
                
                if s==starts+1
                    c;
                end
            end
 %%           
            
            %----------PCA of residuals-------------------
            fitted_res=zeros(length(INoise),1);
            
            if isempty(intersect(exc,c)) & npc~=0
                Ipca=filtfilt(hpfwts,1,double(Idata-INoise));
                pccount=1;
                skcount=1;
                pick=cumsum(ones(markerl(sec),1)*2+round(rand(markerl(sec),1)));
                if strig~=1
                    pick=[pick(1):pick(end)];
                end
                
                
                for s=starts+1:lasts-1
                    % construct PCAMAT
                    if skcount==pick(pccount)
                        pcamat(pccount,:)=...
                            Ipca(secmarker(s)-pre_peak:...
                            secmarker(s)+max_postpeak);
                        pccount=pccount+1;
                    end
                    skcount=skcount+1;
                end
                
                pcamat=detrend(pcamat','constant')';
                [apc,ascore,asvar]=pca_calc(pcamat(1:(pccount-1),:)');
                
                oev=100*asvar/sum(asvar);
                if sec==1
                    if ischar(npc)
                        d_oev=find(abs(diff(oev))<TH_SLOPE);
                        dd_oev=diff(d_oev);
                        for I=1:length(dd_oev)-3
                            if [dd_oev(I) dd_oev(I+1) dd_oev(I+2)]==[1 1 1]
                                break
                            end
                        end
                        SLOPETH_PC=d_oev(I)-1;
                        TMPTH=find(cumsum(oev)>TH_CUMVAR);
                        CUMVARTH_PC=TMPTH(1);
                        TMPTH=find(oev<TH_VAREXP);
                        VAREXPTH_PC=TMPTH(1)-1;
                        pcs=floor(mean([SLOPETH_PC CUMVARTH_PC VAREXPTH_PC]));
                        fprintf('\n%d residual PCs will be removed from channel %d\n . If you get an error "line 746 of fmrib_fastr: index exceeds matrix dimensions" it means there is an inconsistency in your TR triggers, either the TR length or the number of markers',pcs,c);
                    else
                        pcs=npc;
                    end
                end
                
                % TEST CODE
                %             SPCS(c)=SLOPETH_PC;
                %             CPCS(c)=CUMVARTH_PC;
                %             VPCS(c)=VAREXPTH_PC;
                %             PCS(c)=pcs;
                
                
                if strig==0
                    papc=double([ascore(:,1:pcs) ones(pre_peak+max_postpeak+1,1)]);
                else
                    papc=double([ascore(:,1:pcs)]);
                end
                
                
                minmax1=max(papc(:,1))-min(papc(:,1));
                for apc=2:pcs
                    papc(:,apc)=papc(:,apc)*minmax1/...
                        (max(papc(:,apc))-min(papc(:,apc)));
                end
                
                for s=starts:lasts
                    if s==1
                        if ~STARTFLAG
                            fitted_res(secmarker(s)-pre_peak:secmarker(s)+max_postpeak)=...
                                papc*(papc\...
                                double(Ipca(secmarker(s)-pre_peak:...
                                secmarker(s)+max_postpeak))');
                        end
                    elseif s==lasts & sec==sections
                        if ~LASTFLAG
                            fitted_res(secmarker(s)-pre_peak:secmarker(s)+max_postpeak)=...
                                papc*(papc\...
                                double(Ipca(secmarker(s)-pre_peak:...
                                secmarker(s)+max_postpeak))');
                        end
                    else
                        fitted_res(secmarker(s)-pre_peak:secmarker(s)+max_postpeak)=...
                            papc*(papc\...
                            double(Ipca(secmarker(s)-pre_peak:...
                            secmarker(s)+max_postpeak))');
                    end
                end
                
            elseif strig==0 % not doing OBS and using volume triggers
                
                Ipca=Idata-INoise;
                papc=double(ones(pre_peak+max_postpeak+1,1));
                for s=starts:lasts
                    if s==1
                        if ~STARTFLAG
                            fitted_res(secmarker(s)-pre_peak:secmarker(s)+max_postpeak)=...
                                papc*(papc\...
                                double(Ipca(secmarker(s)-pre_peak:...
                                secmarker(s)+max_postpeak))');
                        end
                    elseif s==lasts & sec==sections
                        if ~LASTFLAG
                            fitted_res(secmarker(s)-pre_peak:secmarker(s)+max_postpeak)=...
                                papc*(papc\...
                                double(Ipca(secmarker(s)-pre_peak:...
                                secmarker(s)+max_postpeak))');
                        end
                    else
                        fitted_res(secmarker(s)-pre_peak:secmarker(s)+max_postpeak)=...
                            papc*(papc\...
                            double(Ipca(secmarker(s)-pre_peak:...
                            secmarker(s)+max_postpeak))');
                    end
                end
                
            end
            
            %-----------------end PCA Section------------------
            
            Idata=Iorig-INoise-fitted_res';
            
            if L > 1
                fcleanEEG=decimate2(Idata,L);
                fNoise=decimate2(INoise+fitted_res',L);
            else
                fcleanEEG=Idata;
                fNoise=INoise+fitted_res';
            end
            
            
            if sec==1
                if sections==1
                    cleanEEG(d1:d2)=fcleanEEG;
                    Noise(d1:d2)=fNoise;
                else
                    cleanEEG(d1:(d1-1)+secl(sec))=fcleanEEG(1:end-pad_sec);
                    Noise(d1:(d1-1)+secl(sec))=fNoise(1:end-pad_sec);
                end
            elseif sec==sections
                cleanEEG((d1-1)+sum(secl(1:sec-1))+1:d2)=...
                    fcleanEEG(pad_sec+1:end);
                Noise((d1-1)+sum(secl(1:sec-1))+1:d2)=fNoise(pad_sec+1:end);
            else
                cleanEEG((d1-1)+sum(secl(1:sec-1))+1:(d1-1)+sum(secl(1:sec)))=...
                    fcleanEEG(pad_sec+1:end-pad_sec);
                Noise((d1-1)+sum(secl(1:sec-1))+1:(d1-1)+sum(secl(1:sec)))=...
                    fNoise(pad_sec+1:end-pad_sec);
            end
            
            
            %Update progress bar
%             scount=scount+1;
%             percentdone=floor(scount*100/steps);
%             if floor(percentdone)>=barth
%                 
%                 if percentdone>=25 & Flag25==0
%                     fprintf('25%% ')
%                     Flag25=1;
%                 elseif percentdone>=50 & Flag50==0
%                     fprintf('50%% ')
%                     Flag50=1;
%                 elseif percentdone>=75 & Flag75==0
%                     fprintf('75%% ')
%                     Flag75=1;
%                 elseif percentdone==100
%                     fprintf('100%%\n')
%                 else
%                     fprintf('.')
%                 end
%                 
%                 while barth<=percentdone
%                     barth=barth+barth_step;
%                 end
%                 if barth>100
%                     barth=100;
%                 end
%            end
        end
        
        if lpf>0
            cleanEEG=filtfilt(lpfwts,1,double(cleanEEG));
            Noise=filtfilt(lpfwts,1,double(Noise));
        end
        
        if (anc_chk==1) && isempty(intersect(exc,c))
            % Adaptive Noise cancellation
            % ---------------------------
            refs=Noise(d1:d2)';
            tmpd=filtfilt(ANCfwts,1,double(cleanEEG))';
            d=double(tmpd(d1:d2));
            Alpha=sum(d.*refs)/sum(refs.*refs);
            refs=double(Alpha*refs);
            mu=double(0.05/(N*var(refs)));
            [out,y]=fastranc(refs,d,N,mu);
            if isinf(max(y)) || any(isnan(y))
                wst=sprintf('ANC Failed for channel number %d. Skipping ANC.',c);
                warning(wst);
                EEGData{c}(d1:d2) = cleanEEG(d1:d2);
            else
                EEGData{c}(d1:d2)=cleanEEG(d1:d2)-y';
            end
        else
            EEGData{c}(d1:d2)=cleanEEG(d1:d2);
        end
        
%         scount=scount+1;
%         percentdone=floor(scount*100/steps);
%         if floor(percentdone)>=barth
%             
%             if percentdone>=25 & Flag25==0
%                 fprintf('25%% ')
%                 Flag25=1;
%             elseif percentdone>=50 & Flag50==0
%                 fprintf('50%% ')
%                 Flag50=1;
%             elseif percentdone>=75 & Flag75==0
%                 fprintf('75%% ')
%                 Flag75=1;
%             elseif percentdone==100
%                 fprintf('100%%\n')
%             else
%                 fprintf('.')
%             end
%             
%             while barth<=percentdone
%                 barth=barth+barth_step;
%             end
%             if barth>100
%                 barth=100;
%             end
%         end
    end
catch
    keyboard
end;

% Convert the parallelized variables into regular variables:
EEG.data = cell2mat(EEGData);

% Calculate processing time
% --------------------------

mttoc=floor(toc/60);
sttoc=round(toc-mttoc*60);
if mttoc < 60
    fprintf('FASTR Finished in %d min %d sec.\n',mttoc,sttoc);
else
    httoc=floor(mttoc/60);
    mttoc=round(mttoc-httoc*60);
    fprintf('FASTR Finished in %d hrs %d min %d sec.\n',httoc,mttoc,sttoc);
end
return;
