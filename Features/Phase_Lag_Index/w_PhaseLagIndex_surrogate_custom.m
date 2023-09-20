function surro_WPLI = w_PhaseLagIndex_surrogate_custom(X)
% Given a multivariate data, returns phase lag index matrix
% Modified the mfile of 'phase synchronization'
ch = size(X,2); % column should be channel
splice = randi(size(X,1));  % determines random place in signal where it will be spliced

a_sig = X;
a_sig2 = [a_sig(splice:size(a_sig,1),:); a_sig(1:splice-1,:)];  % %This is the randomized signal
surro_WPLI=ones(ch,ch);

% More efficient:
for ch1 = 1:ch
    d1 = repmat(a_sig(:,ch1),[1,ch]);    
    c_sig = d1.*conj(a_sig2);
    
    numer = abs(mean(imag(c_sig))); % average of imaginary
    denom = mean(abs(imag(c_sig))); % average of abs of imaginary
    
    numer(ch1) = 1; denom(ch1) = 1;
    
    surro_WPLI(ch1,:) = numer./denom;    
end

surro_WPLI = surro_WPLI - eye(ch); % Remove 1s along diagonal
surro_WPLI = triu(surro_WPLI) + triu(surro_WPLI)' - eye(size(surro_WPLI,1)).*surro_WPLI; % Make it symmetric


% Original Less efficient:
% for c1=1:ch-1
%     for c2=c1+1:ch
%         c_sig=a_sig(:,c1).*conj(a_sig2(:,c2));
%         
%         numer=abs(mean(imag(c_sig))); % average of imaginary
%         denom=mean(abs(imag(c_sig))); % average of abs of imaginary
%         
%         surro_WPLI(c1,c2)=numer/denom;
%         surro_WPLI(c2,c1)=surro_WPLI(c1,c2);
%     end
% end 

