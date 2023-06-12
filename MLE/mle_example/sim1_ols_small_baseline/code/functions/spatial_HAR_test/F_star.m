%% function to construct centered orthonormal basis F* test

function [F_trans,data_test] = F_star(R,r,ydep,x_reg,loc,K1,K2)

    xloc = loc(:,1);
    yloc = loc(:,2);

    p = size(R,2);   %% number of regressors
    q = size(R,1);          %% number of restrictions
        
    N = size(xloc,1);

    nx = max(xloc)-min(xloc); 
    ny = max(yloc)-min(yloc);
        
    K = K1 + K2 + K1*K2;
    
    theta_hat = x_reg\ydep;
    iH = (x_reg'*x_reg/N)\eye(p,p);

    e_hat= ydep-x_reg*theta_hat ;
    xe_hat_all = repmat(e_hat,1,p).*x_reg;
    
    ave_phi_k = zeros(K1+1,K2+1);

    for k1 = 1 : K1+1  %% To obtain the average of basis function
      for k2 = 1 : K2+1
        ave_phi_k(k1,k2) = sum(exp(-1i*2*pi*((k1-1)*xloc/nx + (k2-1)*yloc/ny)))/N;
      end
    end

    temp = zeros(1,0.5*(K1 + K2)*(K1 + K2 + 1)+K2 + 1);  %% Cantor pairing function (To determine the order of bases, I used Cantor pairing function) 
    phi_trans = zeros(N,0.5*(K1 + K2)*(K1 + K2 + 1)+K2 + 1);

    for k1 = 0 : K1
      for k2 = 0 : K2
        phi_trans(:,0.5*(k1+k2)*(k1+k2+1)+k2+1) = exp(-1i*2*pi*(k1*xloc/nx + k2*yloc/ny)) - ave_phi_k(k1+1,k2+1);      %% center the basis function
        temp(1,0.5*(k1+k2)*(k1+k2+1)+k2+1) = 1;
      end
    end

    phi_trans(:,temp==0)=[];

    real_phi_trans = real(phi_trans(:,2:K+1));
    imag_phi_trans = imag(phi_trans(:,2:K+1));

    [Q,RR] = GS([real_phi_trans imag_phi_trans]);         %% GS procedure

    phi_OC = Q*sqrt(N);
    
    dft_all_trans = zeros(2*K,p);

    for jj = 1 : p;
      dft_all_trans(:,jj) = sum(phi_OC.*repmat(xe_hat_all(:,jj),1,2*K))';
    end;

    Omega_hat_trans = real(dft_all_trans'*dft_all_trans)/(2*N*K);
    Omega_hat_trans = iH*Omega_hat_trans*iH;
    
    IROR_trans = (R*Omega_hat_trans*R')\eye(q);
    
    F_trans = (2*K-q+1)/(2*K)*(sqrt(N)*(R*theta_hat-r))'*IROR_trans*(sqrt(N)*(R*theta_hat-r))/q; 
    
    % data used for power calculation
    se = sqrt(R*Omega_hat_trans*R')/sqrt(length(ydep));
    data_test = {theta_hat(1),se};
    
    
    
    
    
    
    
    
    
    
