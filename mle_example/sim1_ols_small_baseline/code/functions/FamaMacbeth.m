function [b,se,btemp] = FamaMacbeth(D,X,Y,Z,index)
% run Fama-MacBeth given a clustering structure. Returns FM estimator,
% standard error, and with-in group estimators. 

p = numel(unique(index));
X_mat = [D,X];
Z_mat = [Z,X];
k = size(X_mat,2);
clusters = unique(index);
G = numel(clusters);

btemp = zeros(p,k);
pmiss = 0;
for ii = 1:G
    fii = index == ii;
    if sum(fii) > k
        btemp(ii,:) = (Z_mat(fii,:)'*X_mat(fii,:))\...
            (Z_mat(fii,:)'*Y(fii,:));
    else
        btemp(ii,:) = NaN*ones(1,k);
        pmiss = pmiss+1;
    end
end

b = nanmean(btemp);
se = nanstd(btemp)/sqrt(p-pmiss);