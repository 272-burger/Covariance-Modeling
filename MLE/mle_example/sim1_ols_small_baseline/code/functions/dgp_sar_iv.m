function [U,V] = dgp_sar_iv(N,T,dis_mat_unique)

rho = exp(-1); % intertemporal correlation
theta = .15; % spatial correlation
d_bar = .3; % weight matrix threshold
W = dis_mat_unique<d_bar;
W = W-diag(diag(W)); % SAR weight matrix 
n = N*T;

iidError = mvnrnd([0 0],[1 .8 ; .8 1],n);

% error in structual equation
invSAR = eye(N)-theta*W;
U1 = invSAR\iidError(1:N,1);
U2 = rho*U1+sqrt(1-rho^2)*(invSAR\iidError(N+1:end,1));
U = zeros(N*T,1);
U(1:2:N*T) = U1;
U(2:2:N*T) = U2;

% error in first-stage equation
V1 = invSAR\iidError(1:N,2);
V2 = rho*V1+sqrt(1-rho^2)*(invSAR\iidError(N+1:end,2));
V = zeros(N*T,1);
V(1:2:N*T) = V1;
V(2:2:N*T) = V2;
