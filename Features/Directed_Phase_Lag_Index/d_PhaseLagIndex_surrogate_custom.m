function PLI = d_PhaseLagIndex_surrogate_custom(X)
% Given a multivariate data, returns phase lag index matrix
% Modified the mfile of 'phase synchronization'
% PLI(ch1, ch2) : 
% if it is greater than 0.5, ch1->ch2
% if it is less than 0.5, ch2->ch1

ch=size(X,2); % column should be channel
splice = randi(size(X,1));  % determines random place in signal where it will be spliced

%%%%%% Hilbert transform and computation of phases
phi1 = angle(X);
phi2 = [phi1(splice:size(phi1,1),:); phi1(1:splice-1,:)];  % %This is the randomized signal
% for i=1:ch
%     x=X(:,i);
%     %     phi0=angle(hilbert(x));  % only the phase component
%     %     phi1(:,i)=unwrap(phi0);  % smoothing
%     phi1(:,i)=angle(hilbert(x));
%     phi2(:,i) = [phi1(splice:size(phi1,1),i); phi1(1:splice-1,i)];  % %This is the randomized signal
% end

PLI = ones(ch,ch);

% More efficient:
for ch1 = 1:ch
    d1 = repmat(phi1(:,ch1),[1,ch]);    
    PDiff = d1 - phi2;
    
    PLI(ch1,:) = mean(heaviside(sin(PDiff)));   
end

% % Original less efficient:
% for ch1=1:ch
%     for ch2=1:ch
%         %%%%%% phase lage index
%         PDiff=phi1(:,ch1)-phi2(:,ch2); % phase difference
% %         PLI(ch1,ch2)=mean(sign(PDiff)); % only count the asymmetry
%         PLI(ch1,ch2)=mean(heaviside(sin(PDiff)));
%     end
% end

% By definition,
% if PLI(ch1,ch2) is greater than 0.5, ch1 is leading ch2.
% if it is less than 0.5, ch1 is lagged by ch2.



