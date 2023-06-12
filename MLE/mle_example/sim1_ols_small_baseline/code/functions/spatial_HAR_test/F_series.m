%% function to construct F statistic

function F_simulation = F_series(R,r,ydep,x_reg,loc,K1,K2)

    xloc = loc(:,1);
    yloc = loc(:,2);

    p = size(R,2);    %% number of regressors
    q = size(R,1);           %% number of restrictions
    
    N = size(x_reg,1);       %% sample size
    
    nx = max(xloc)-min(xloc); 
    ny = max(yloc)-min(yloc);

    K = K1 + K2 + K1*K2;
    
    theta_hat = x_reg\ydep;
    iH = (x_reg'*x_reg/N)\eye(p,p);

    e_hat= ydep-x_reg*theta_hat ;
    xe_hat_all = repmat(e_hat,1,p).*x_reg;

    dft_xe_all = zeros(K+1,p);

    for jj = 1 : p;
        DFT_xe = zeros(K1+1,K2+1);
        for k1 = 1 : K1+1;
            for k2 = 1 : K2+1;
                DFT_xe(k1,k2) = sum(sum(exp(-1i*2*pi*((k1-1)*xloc/nx+(k2-1)*yloc/ny))'*xe_hat_all(:,jj)));
            end;
        end;
        dft_xe_all(:,jj) =  DFT_xe(:);
    end;
    
    Omega_hat = real(dft_xe_all'*dft_xe_all)/(K*N);
    Omega_hat = iH*Omega_hat*iH;
    
    IROR = (R*Omega_hat*R')\eye(q);
    
    F_simulation = (sqrt(N)*(R*theta_hat-r))'*IROR*(sqrt(N)*(R*theta_hat-r))/q; 