

rng(91720)
restoredefaultpath % delete paths to avoid conflict
addpath('functions')
addpath('functions/spatial_HAR_test')
if testMode == 1
    B = 4;
    Bboot = 10;
end


%% load location data from empirical example
if sampleScale == "small"
    preload_data_small
    pre_load = load('../temp/preload_data_small.mat');
elseif sampleScale == "large"
    preload_data_large
    pre_load = load('../temp/preload_data_large.mat');
end
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

% extra covariates
if lower(modelName) == "ols"
    beta0 = zeros(p+1,1);
else
    pi0 = 2;
    betaD = zeros(p+1,1);
    betaY = zeros(p+1,1);
end

% alternatives reported in power curves 
if testMode == 0
    altPowerCurve = -4:.01:4;
else
    altPowerCurve = -4:.5:4;
end
nAltPowerCurve = numel(altPowerCurve);

% alternatives considered in parametric bootstrap
altPowerSim = (-10:1:10)'/sqrt(n);
altPowerSim(11) = [];
nAltPowerSim = numel(altPowerSim);

% pre-loaded clusterings
group_location = pre_load.group_location; % cluster by location
group_matrix_km = pre_load.group_matrix_km(:,1:G_bar-1); % k-medoids

% pre-calculate conditional heteroskedasticity
if lower(spatialModel) == "hetero"
    coord = pre_load.coord;
    temp = coord-repmat(mean(coord),length(coord),1);
    hetero = abs(sqrt(sum(temp.^2,2)));
end

                    
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

if lower(modelName) == "ols"
    D = regressor_matrix(:,1);
    X = [regressor_matrix(:,2:end),ones(n,1)];
    mD = D - X*(X\D);
    MDX = eye(n) - ([X D]/([X D]'*[X D]))*[X D]';
    [~,~,ex] = qr(MDX,'vector');
    useQML = ex(1:end-p-2);
else
    Z = regressor_matrix(:,1);
    X = [regressor_matrix(:,2:end),ones(n,1)];
    mZ = Z - X*(X\Z);
    % Only going to account for partialling out controls, not instrument or
    % worry about endogenous variable
    M = eye(n) - X*((X'*X)\X');
    [~,~,ex] = qr(M,'vector');
    useQML = ex(1:end-p-1);
end


%% define result matrices

if lower(modelName) == "ols"
    qmle = zeros(B,3);
else
    qmle = zeros(B,7);
end

% CCE
ahat = zeros(B,1); % full sample least sequares estimator
seLocation = zeros(B,1);
cvalLocation = zeros(B,1);
seG_cce = zeros(B,l_G);
cvalG_cce = zeros(B,l_G);
seGstarA = zeros(B,1);
cvalGstarA = zeros(B,1);
GstarA = zeros(B,1);
seGstarB = zeros(B,1);
cvalGstarB = zeros(B,1);
GstarB = zeros(B,1);

% IM
ahatG_fm = zeros(B,l_G);
seG_im = zeros(B,l_G);
cvalG_im = zeros(B,l_G);
aGstar_im = zeros(B,1);
seGstar_im = zeros(B,1);
cvalGstar_im = zeros(B,1);
Gstar_im = zeros(B,1);

% CRS
pValNullG_crs = zeros(B,l_G);
pValAdjG_crs = zeros(B,l_G);
pValPowerCurveG_crs = zeros(B,nAltPowerCurve,l_G);
pValPowerCurveGstar_crs = zeros(B,nAltPowerCurve);
pValGstar_crs = zeros(B,1);
pvalAdjGstar_crs = zeros(B,1);
Gstar_crs = zeros(B,1);
aGstar_crs = zeros(B,1);

% SK
pValNull_sk = zeros(B,1);
pValAlt_sk = zeros(B,nAltPowerCurve);


