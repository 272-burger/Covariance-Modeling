

load('../temp/all_results.mat')


% index of power grid in summary table
indPowerGrid = [find(altPowerCurve == -1),...
    find(altPowerCurve == -.5),find(altPowerCurve == .5),...
    find(altPowerCurve == 1)];


%% CCE by location
if lower(modelName) == "ols"
    mean_loc = mean(ahat);
    rmse_loc = sqrt(mean((ahat-alpha0).^2));
else
    mean_loc = median(ahat);
    rmse_loc = median(abs(ahat-alpha0));
end

size_loc_u = mean(abs(ahat./seLocation) > tinv(1-sigLevel/2,max(...
    group_location)-1)); % usual critical value
size_loc = mean(abs(ahat./seLocation) > cvalLocation);

powerCurve_loc_u = zeros(nAltPowerCurve,1);
powerCurve_loc = zeros(nAltPowerCurve,1);
for jj = 1 : nAltPowerCurve
    powerCurve_loc_u(jj) = mean(abs((ahat-altPowerCurve(jj))./...
        seLocation) > tinv(1-sigLevel/2,max(group_location)-1));
    powerCurve_loc(jj) = mean(abs((ahat-altPowerCurve(jj))./...
        seLocation) > cvalLocation);
end
powerGrid_loc_u = powerCurve_loc_u(indPowerGrid)';
powerGrid_loc = powerCurve_loc(indPowerGrid)';


%% CCE
% A = including cluster by location as one candidate clustering
% B = excluding cluster by location
sizeReg_cce = mean(abs(repmat(ahat,1,l_G)./seG_cce) > tinv(1-sigLevel/2,...
    max(group_matrix_km)-1));
sizeG_cce = mean(abs(ahat./seG_cce) > cvalG_cce);
sizeGstarA_cce = mean(abs(ahat./seGstarA) > cvalGstarA);
sizeGstarB_cce = mean(abs(ahat./seGstarB) > cvalGstarB);
    
powerCurve_cceG = zeros(nAltPowerCurve,l_G);
powerCurve_cceGstarA = zeros(nAltPowerCurve,1);
powerCurve_cceGstarB = zeros(nAltPowerCurve,1);
for jj = 1:numel(altPowerCurve)
    for kk = 1:l_G
        powerCurve_cceG(jj,kk) = mean(abs((ahat-altPowerCurve(jj))./...
            seG_cce(:,kk)) > cvalG_cce(:,kk));
    end
    powerCurve_cceGstarA(jj) = mean(abs((ahat-altPowerCurve(jj))./...
        seGstarA) > cvalGstarA);
    powerCurve_cceGstarB(jj) = mean(abs((ahat-altPowerCurve(jj))./...
        seGstarB) > cvalGstarB);
end
powerGrid_cce = powerCurve_cceGstarB(indPowerGrid)';

G_list = [2:G_bar,N];
GdistA_cce = mean(repmat(G_list,B,1) == repmat(GstarA,1,length(G_list)));
G_list = 2:G_bar;
GdistB_cce = mean(repmat(G_list,B,1) == repmat(GstarB,1,length(G_list)));


%% IM
sizeReg_im = mean(abs(ahatG_fm./seG_im) > tinv(1-sigLevel/2,...
    max(group_matrix_km)-1));
sizeG_im = mean(abs(ahatG_fm./seG_im) > cvalG_im);
if lower(modelName) == "ols"
    mean_im = mean(aGstar_im);
    rmse_im = sqrt(mean(aGstar_im.^2));
else
    mean_im = median(aGstar_im);
    rmse_im = median(abs(aGstar_im-alpha0));
end
sizeGstar_im = mean(abs(aGstar_im./seGstar_im) > cvalGstar_im);

powerCurveG_im = zeros(nAltPowerCurve,l_G);
powerCurveGstar_im = zeros(nAltPowerCurve,1);
for jj = 1 : nAltPowerCurve
    for kk = 1:l_G
        powerCurveG_im(jj,kk) = mean(abs((ahatG_fm(:,kk)-...
            altPowerCurve(jj))./seG_im(:,kk)) > cvalG_im(:,kk));
    end
    powerCurveGstar_im(jj) = mean(abs((aGstar_im-altPowerCurve(jj))./...
        seGstar_im) > cvalGstar_im);
end
powerGrid_im = powerCurveGstar_im(indPowerGrid)';

G_list = 2:G_bar;
Gdist_im = mean(repmat(G_list,B,1) == repmat(Gstar_im,1,length(G_list)));


%% CRS
sizeReg_crs = mean(pValNullG_crs <= sigLevel);
sizeG_crs = mean(pValNullG_crs <= pValAdjG_crs);
if lower(modelName) == "ols"    
    mean_crs = mean(aGstar_crs);
    rmse_crs = sqrt(mean(aGstar_crs.^2));
else
    mean_crs = median(aGstar_crs);
    rmse_crs = median(abs(aGstar_crs-alpha0));
end
sizeGstar_crs = mean(pValGstar_crs <= pvalAdjGstar_crs);

powerCurveG_crs = zeros(nAltPowerCurve,l_G);
for kk = 1 : l_G
    powerCurveG_crs(:,kk) = mean(pValPowerCurveG_crs(:,:,kk) <= ...
        repmat(pValAdjG_crs(:,kk),1,nAltPowerCurve))';
end
powerCurveGstar_crs = mean(pValPowerCurveGstar_crs <= ...
    repmat(pvalAdjGstar_crs,1,nAltPowerCurve));
powerGrid_crs = powerCurveGstar_crs(indPowerGrid)';
    
G_list = 2:G_bar;
Gdist_crs = mean(repmat(G_list,B,1) == repmat(Gstar_crs,1,length(G_list)));


%% SK
size_sk = mean(pValNull_sk < sigLevel);
powerCurve_sk = mean(pValAlt_sk < sigLevel);
powerGrid_sk = powerCurve_sk(indPowerGrid)';


%% generate all tables and figures
table_simulation_results
table_kHat_distribution
table_alphaHat_distribution
table_summary_complete
table_clustering_complete

figure_power_curves



