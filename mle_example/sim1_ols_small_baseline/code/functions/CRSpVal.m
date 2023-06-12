function [pval] = CRSpVal(betaVec)

G = size(betaVec,1);
signFlipper = 2*(de2bi(0:2^G-1)-.5); % list all transformations

% % use mean as the test statistic
% b_distr = mean(signFlipper.*repmat(betaVec',2^G,1),2);
% pval = mean(abs(mean(betaVec)) <= abs(b_distr));

% use t-stat as the test statistic
betaVec_distr = signFlipper.*repmat(betaVec',2^G,1);
tstat = sqrt(G)*mean(betaVec)/std(betaVec);
tstat_distr = sqrt(G)*mean(betaVec_distr,2)./std(betaVec_distr,0,2);
pval = mean(abs(tstat) <= abs(tstat_distr));