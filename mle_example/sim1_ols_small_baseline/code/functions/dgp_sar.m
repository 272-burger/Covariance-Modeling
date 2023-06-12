function [U] = dgp_sar(N,T,dis_mat_unique)

rho = exp(-1); % intertemporal correlation
theta = .15; % spatial correlation
d_bar = .3; % weight matrix threshold
W = dis_mat_unique<d_bar;
W = W-diag(diag(W)); % SAR weight matrix 

% U
V1 = randn(N,1);
V2 = randn(N,1);
U1 = (eye(N)-theta*W)\V1;
U2 = rho*U1+sqrt(1-rho^2)*((eye(N)-theta*W)\V2);
U = zeros(N*T,1);
U(1:2:N*T) = U1;
U(2:2:N*T) = U2;



