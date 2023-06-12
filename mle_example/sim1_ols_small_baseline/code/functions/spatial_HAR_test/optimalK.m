%% function to derive data dependent optimal K1 and K2

function [K1,K2]= optimalK(R,ydep,x_reg,xloc,yloc,nu)

p = size(x_reg,2);   %% number of regressors
N = size(x_reg,1);   %% sample size

nx = max(xloc)-min(xloc); 
ny = max(yloc)-min(yloc);
 
theta_hat = x_reg\ydep;  % OLS estimator

iH = (x_reg'*x_reg/N)\eye(p,p); %%

e_hat= ydep-x_reg*theta_hat ;
xe_hat_all = repmat(e_hat,1,p).*x_reg;

de_hat = (R*iH*xe_hat_all')';
q = size(R,1);

c1 = zeros(q,1);
r1 = zeros(q,1);

ntries = 2;

         
% parfor j = 1 : q;
for j = 1 : q;
% 
    for jj = 1 : floor(sqrt(nx^2+ny^2))  
        d1 = variogram([xloc, yloc],de_hat(:,j),'nrbins',jj);
        if sum(isnan(d1.val))>0, bin = jj-1;
            clear d1;
            d1 = variogram([xloc, yloc],de_hat(:,j),'nrbins',bin);
        break, end;
    end
           
    r_temp =zeros(ntries,1);
    c_temp =zeros(ntries,1);
    Rs = zeros(ntries,1);
    
    initial = [1,1;
           rand(ntries, 2)];
       
    for itries =1:ntries+1 
      r0 = min(d1.distance)+initial(itries,1)* (max(d1.distance)-min(d1.distance)); % initial value: range 
      c0 = min(d1.val)+ initial(itries,2)* (max(d1.val)-min(d1.val))  ;             % initial value: sill 
      [r_temp(itries,1),c_temp(itries,1),~,S] = variogramfit(d1.distance,d1.val,r0,c0,[],...
                       'solver','fminsearchbnd',...
                       'model', 'matern',...
                       'nugget',[],...
                       'nu', nu,...
                       'plotit',false);
        Rs(itries)=S.Rs;
    end
    
    [~,ind] = max(Rs);
                   
                   
    r1(j,1) = r_temp(ind);
    c1(j,1) = c_temp(ind);
                   
end;

const = sum((c1.^2).*(r1.^4))*(nu+1)^(-2);
const = const/sum((c1.^2).*(r1.^8));

b = 0.25501*const^(1/6)*(N^(-1/6));


K1 = floor(b*nx);

K1 = min(K1,floor(nx/2));

K1 = min(K1,25);

K2 = floor(b*ny);

K2 = min(K2,floor(ny/2));

K2 = min(K2,25);
