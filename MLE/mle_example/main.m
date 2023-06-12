
%% INITIALIZATION
clear
tic
rng(91720)
restoredefaultpath % delete paths to avoid conflict

% load location data from empirical example
preload_data_small
pre_load = load('preload_data_small.mat');
Xcoord = pre_load.coord(:,1);
Ycoord = pre_load.coord(:,2);
dis_mat_unique = pre_load.dis_mat_unique; % distance metric


%% parameters 
sigLevel = .05;
N = size(dis_mat_unique,1); % number of locations
T = 2; % number of time periods
n = N*T; % sample size
p = 10; % number of extra covariates
theta = 3; % spatial correlation
rho = 1; % inter-temporal correlation
rho_regressor = .5; % covariance among covariates
alpha0 = 0; % parameter of interest
G_bar = ceil(n^(1/3)); % maximal number of clusters considered
G_vec = 2:G_bar; % candidate numbers of clusters
l_G = numel(G_vec); % number of candidate clusterings
beta0 = zeros(p+1,1);

% alternatives considered in parametric bootstrap
altPowerSim = (-10:1:10)'/sqrt(n);
altPowerSim(11) = [];
nAltPowerSim = numel(altPowerSim);

% pre-loaded clusterings
group_location = pre_load.group_location; % cluster by location
group_matrix_km = pre_load.group_matrix_km(:,1:G_bar-1); % k-medoids


%% covariance matrices
dis_mat = pre_load.dis_mat;
time_mat = squareform(pdist(kron(ones(N,1),[1;2])));
Sigma = exp(-dis_mat/theta-time_mat/rho);
% Sigma = eye(n); % sanity check
CSigma = chol(Sigma);
Sigma_control = (rho_regressor*ones(1+p)+(1-...
    rho_regressor)*eye(1+p))/2; % covariance between regressors


%% draw one realization of design variables. Condition on X
regressor_matrix = CSigma'*randn(n,1+p)*...
    chol(Sigma_control);

D = regressor_matrix(:,1);
X = [regressor_matrix(:,2:end),ones(n,1)];
mD = D - X*(X\D);
MDX = eye(n) - ([X D]/([X D]'*[X D]))*[X D]';
[~,~,ex] = qr(MDX,'vector');
useQML = ex(1:end-p-2);


%% DGP

U = CSigma'*randn(n,1);

Y = D*alpha0+X*beta0+U; % generate outcome for this realization
mY = Y - X*(X\Y); % full sample least squares coefficient
ahat = mD\mY; % full sample least sequares estimator
resid = mY-mD*ahat; % residual


%% COVARIANCE ESTIMATION

% estimate homogeneous exponential covariance model  
Sigma_func = @(w) MDX(useQML,:)*(exp(w(1))*exp(-dis_mat/exp(w(2))- ...
    time_mat/exp(w(3))))*(MDX(useQML,:)');
Q = @(w) .5*logdet(Sigma_func(w))+.5*resid(useQML,1)'*...
    (Sigma_func(w)\resid(useQML,1));
options = optimoptions('fminunc','display','off');
alpha_init = [0;0;0];
alphaHat = fminunc(Q,alpha_init,options);

% generate covariance matrix estimate
Sigma_func_DGP = @(w) exp(w(1))*exp(-dis_mat/exp(w(2))- ...
    time_mat/exp(w(3)));
SigmaHat = Sigma_func_DGP(alphaHat); % estimated covariance matrix

% psd check 
if min(eig(SigmaHat)) < 0
    error('Non-psd covariance matrix');
end




toc
