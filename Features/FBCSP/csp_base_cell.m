function P = csp_base_cell(X,Y,opt)
% Computes the base Common Spatial Patterns filters using one of three
% methods. This version operates on cell arrays
% Written by Kiret Dhindsa

nTrials = length(X);
nChans = size(X{1},2);
ys = unique(Y);
switch opt.mode
    case 'concat'
        % Concatenate trials by class
        X1 = cat(1,X{Y==ys(1)});
        X2 = cat(1,X{Y==ys(2)});
        
        % Compute Covariance Matrices
        R1 = cov(X1);
        R2 = cov(X2);
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
        % Concatenate trials by class
        X1 = cat(1,X{Y==ys(1)});
        X2 = cat(1,X{Y==ys(2)});
        
        % Compute Covariance Matrices
        R1 = cov(X1);
        R2 = cov(X2);
        Rc = R1 + R2;
        
        % Generalized eigenvector decomposition
        [V,~] = eig(R1,Rc);
%         P = inv(V);
        P = V(:,[1:opt.m,end-opt.features.cspcomps+1:end])';
        
    case 'trials'
        % Covariance matrix for each trial
        R = zeros(nChans,nChans,nTrials);
        for i = 1:nTrials
            R(:,:,i) = squeeze(X{i})*squeeze(X{i})';
            R(:,:,i) = R(:,:,i)/trace(R(:,:,i));
        end
        
        % Class-averaged covariance matrices and composite covariance matrix
        R1 = mean(R(:,:,Y==ys(1)),3);
        R2 = mean(R(:,:,Y==ys(2)),3);
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