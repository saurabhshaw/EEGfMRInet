function WPLI = w_PhaseLagIndex_custom(bdata)
% INPUT:
%   bdata: band-pass filtered hilbert data

ch = size(bdata,2); % column should be channel
a_sig = bdata;
WPLI = ones(ch,ch);

% % More efficient:
for ch1 = 1:ch
    d1 = repmat(a_sig(:,ch1),[1,ch]);    
    c_sig = d1.*conj(a_sig);
    
    numer = abs(mean(imag(c_sig))); % average of imaginary
    denom = mean(abs(imag(c_sig))); % average of abs of imaginary
    
    numer(ch1) = 1; denom(ch1) = 1;
    
    WPLI(ch1,:) = numer./denom;    
end

WPLI = WPLI - eye(ch); % Remove 1s along diagonal
WPLI = triu(WPLI) + triu(WPLI)' - eye(size(WPLI,1)).*WPLI; % Make it symmetric


% Original Less efficient:
% for c1=1:ch-1
%     for c2=c1+1:ch
%         c_sig=a_sig(:,c1).*conj(a_sig(:,c2));
%         
%         numer=abs(mean(imag(c_sig))); % average of imaginary
%         denom=mean(abs(imag(c_sig))); % average of abs of imaginary
%         
%         WPLI(c1,c2)=numer/denom;
%         WPLI(c2,c1)=WPLI(c1,c2);
%     end
% end 