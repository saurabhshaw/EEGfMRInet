function P = csp_base(X,Y,opt)
% Computes the base Common Spatial Patterns filters using one of three
% methods. 
% Written by Kiret Dhindsa


[nChans,~,nTrials] = size(X);
ys = unique(Y);
switch opt.mode
    case 'concat'
        % Separate trials by class
        X1 = X(:,:,Y==ys(1));
        X2 = X(:,:,Y==ys(2));
        
        % Concatenate trials into long vector
        X1cat = reshape(X1,nChans,[]);
        X2cat = reshape(X2,nChans,[]);
        
        % Compute Covariance Matrices
        R1 = cov(X1cat');
        R2 = cov(X2cat');
        Rc = R1 + R2;
        
        % Eigenvalues and Eigenvectors of composite covariance matrix
        [Bc,lamda] = eig(Rc);
        [lamda,index] = sort(diag(lamda),'descend');
        Bc = Bc(:,index);
        
        % Whitening transform
        W = sqrt(pinv(diag(lamda)))*Bc';
        S1 = W*R1*W';
        S2 = W*R2*W';
        
        % Generalized eigenvector/eigenvalues
        [B,D] = eig(S1,S2);
        [D,ind] = sort(diag(D),'descend');
        B = B(:,ind);
        
        % Projection Matrix
        P = (B'*W);
        P = P([1:opt.features.cspcomps,nChans-opt.features.cspcomps+1:nChans],:);
        
    case 'bcilab'
        % Separate trials by class
        X1 = X(:,:,Y==1);
        X2 = X(:,:,Y==2);
        
        % Concatenate trials into long vector
        X1cat = reshape(X1,nChans,[]);
        X2cat = reshape(X2,nChans,[]);
        
        % Compute Covariance Matrices
        R1 = cov(X1cat');
        R2 = cov(X2cat');
        Rc = R1 + R2;
        
        % Generalized eigenvector decomposition
        [V,~] = eig(R1,Rc);
%         P = inv(V);
        P = V(:,[1:opt.m,end-opt.features.cspcomps+1:end])';
        
    case 'trials'
        % Covariance matrix for each trial
        R = zeros(nChans,nChans,nTrials);
        for i = 1:nTrials
            R(:,:,i) = squeeze(X(:,:,i))*squeeze(X(:,:,i))';
            R(:,:,i) = R(:,:,i)/trace(R(:,:,i));
        end
        
        % Class-averaged covariance matrices and composite covariance matrix
        R1 = mean(R(:,:,Y==1),3);
        R2 = mean(R(:,:,Y==2),3);
        Rc = R1 + R2;
        
        % Eigenvalues and Eigenvectors of composite covariance matrix
        [Bc,lamda] = eig(Rc);
        [lamda,index] = sort(diag(lamda),'descend');
        Bc = Bc(:,index);
        
        % Whitening transform
        W = sqrt(pinv(diag(lamda)))*Bc';
        S1 = W*R1*W';
        S2 = W*R2*W';
        
        % Generalized eigenvector/eigenvalues
        [B,D] = eig(S1,S2);
        [D,ind] = sort(diag(D),'descend');
        B = B(:,ind);
        
        % Projection Matrix
        P = (B'*W);
        P = P([1:opt.features.cspcomps,nChans-opt.features.cspcomps+1:nChans],:);
end