

%% clustering table - comparing clusters 
texOutput = '../output/table_clustering_complete.csv';
if exist(texOutput,'file')
    delete(texOutput);
end
f = fopen(texOutput,'a');
fprintf(f,',,k=2');
fprintf(f,' , %d',3:G_bar);
fprintf(f,',kHat\n');

% CCE
oversizeProb_G = mean(cvalG_cce>repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1));
oversizeProb_Ghat = mean(cvalGstarB>sum((repmat(G_list,B,...
    1) == repmat(GstarB,1,...
    length(G_list))).*repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1),2));

if_oversize = cvalG_cce>repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1);
pValTH = 2*(1-tcdf(cvalG_cce,repmat(G_list,B,1)-1));
pValTH = if_oversize.*pValTH+(1-if_oversize).*sigLevel;
alphaHat_q = [quantile(pValTH,.1); quantile(pValTH,.25);...
    quantile(pValTH,.5); quantile(pValTH,.75); quantile(pValTH,.9)];

if_oversize = cvalGstarB>sum((repmat(G_list,B,...
    1) == repmat(GstarB,1,...
    length(G_list))).*repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1),2);
pValTH = 2*(1-tcdf(cvalGstarB,GstarB-1));
pValTH = if_oversize.*pValTH+(1-if_oversize).*sigLevel;
alphaHat_q = [alphaHat_q,quantile(pValTH,[.1,.25,.5,.75,.9])'];

fprintf(f,'CCE , size (usual cv)');
fprintf(f,' , %1.3f ',sizeReg_cce);
fprintf(f,'\n');
fprintf(f,' , size (simulated cv)');
fprintf(f,' , %1.3f ',sizeG_cce);
fprintf(f,' , %1.3f',sizeGstarB_cce);
fprintf(f,'\n');
fprintf(f,' , kHat frequency');
fprintf(f,' , %1.3f',GdistB_cce);
fprintf(f,'\n');
fprintf(f,',p(sim_size>.05)');
fprintf(f,',%1.3f',oversizeProb_G);
fprintf(f,',%1.3f\n',oversizeProb_Ghat);
fprintf(f,',alphaHat quantile\n');
fprintf(f,',q10');
fprintf(f,',%1.3f',alphaHat_q(1,:));
fprintf(f,'\n');
fprintf(f,',q25');
fprintf(f,',%1.3f',alphaHat_q(2,:));
fprintf(f,'\n');
fprintf(f,',q50');
fprintf(f,',%1.3f',alphaHat_q(3,:));
fprintf(f,'\n');
fprintf(f,',q75');
fprintf(f,',%1.3f',alphaHat_q(4,:));
fprintf(f,'\n');
fprintf(f,',q90');
fprintf(f,',%1.3f',alphaHat_q(5,:));
fprintf(f,'\n');

% IM 
oversizeProb_G = mean(cvalG_im>repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1));
oversizeProb_Ghat = mean(cvalGstar_im>sum((repmat(G_list,B,...
    1) == repmat(Gstar_im,1,...
    length(G_list))).*repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1),2));
 
if_oversize = cvalG_im>repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1);
pValTH = 2*(1-tcdf(cvalG_im,repmat(G_list,B,1)-1));
pValTH = if_oversize.*pValTH+(1-if_oversize).*sigLevel;
alphaHat_q = [quantile(pValTH,.1); quantile(pValTH,.25);...
    quantile(pValTH,.5); quantile(pValTH,.75); quantile(pValTH,.9)];

if_oversize = cvalGstar_im>sum((repmat(G_list,B,...
    1) == repmat(Gstar_im,1,...
    length(G_list))).*repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1),2);
pValTH = 2*(1-tcdf(cvalGstar_im,Gstar_im-1));
pValTH = if_oversize.*pValTH+(1-if_oversize).*sigLevel;
alphaHat_q = [alphaHat_q,quantile(pValTH,[.1,.25,.5,.75,.9])'];

fprintf(f,'IM , size (usual cv)');
fprintf(f,' , %1.3f ',sizeReg_im);
fprintf(f,'\n');
fprintf(f,' , size (simulated cv)');
fprintf(f,' , %1.3f ',sizeG_im);
fprintf(f,' , %1.3f',sizeGstar_im);
fprintf(f,'\n');
fprintf(f,' , kHat frequency');
fprintf(f,' , %1.3f',Gdist_im);
fprintf(f,'\n');
fprintf(f,',p(sim_size>.05)');
fprintf(f,',%1.3f',oversizeProb_G);
fprintf(f,',%1.3f\n',oversizeProb_Ghat);
fprintf(f,',alphaHat quantile\n');
fprintf(f,',q10');
fprintf(f,',%1.3f',alphaHat_q(1,:));
fprintf(f,'\n');
fprintf(f,',q25');
fprintf(f,',%1.3f',alphaHat_q(2,:));
fprintf(f,'\n');
fprintf(f,',q50');
fprintf(f,',%1.3f',alphaHat_q(3,:));
fprintf(f,'\n');
fprintf(f,',q75');
fprintf(f,',%1.3f',alphaHat_q(4,:));
fprintf(f,'\n');
fprintf(f,',q90');
fprintf(f,',%1.3f',alphaHat_q(5,:));
fprintf(f,'\n');

% CRS 
oversizeProb_G = mean(pValAdjG_crs<.05);
oversizeProb_Ghat = mean(pvalAdjGstar_crs<.05);
q10_G = quantile(pValAdjG_crs,.1);
q10_Ghat = quantile(pvalAdjGstar_crs,.1);
q25_G = quantile(pValAdjG_crs,.25);
q25_Ghat = quantile(pvalAdjGstar_crs,.25);
q50_G = quantile(pValAdjG_crs,.50);
q50_Ghat = quantile(pvalAdjGstar_crs,.5);
q75_G = quantile(pValAdjG_crs,.75);
q75_Ghat = quantile(pvalAdjGstar_crs,.75);
q90_G = quantile(pValAdjG_crs,.9);
q90_Ghat = quantile(pvalAdjGstar_crs,.9);

fprintf(f,'CRS , size (usual cv)');
fprintf(f,' , %1.3f ',sizeReg_crs);
fprintf(f,'\n');
fprintf(f,' , size (simulated cv)');
fprintf(f,' , %1.3f ',sizeG_crs);
fprintf(f,', %1.3f',sizeGstar_crs);
fprintf(f,'\n');
fprintf(f,' , kHat frequency');
fprintf(f,' , %1.3f',Gdist_crs);
fprintf(f,'\n');
fprintf(f,',p(sim_size>.05)');
fprintf(f,',%1.3f',oversizeProb_G);
fprintf(f,',%1.3f\n',oversizeProb_Ghat);
fprintf(f,',alphaHat quantile\n');
fprintf(f,',q10');
fprintf(f,',%1.3f',q10_G);
fprintf(f,',%1.3f\n',q10_Ghat);
fprintf(f,',q25');
fprintf(f,',%1.3f',q25_G);
fprintf(f,',%1.3f\n',q25_Ghat);
fprintf(f,',q50');
fprintf(f,',%1.3f',q50_G);
fprintf(f,',%1.3f\n',q50_Ghat);
fprintf(f,',q75');
fprintf(f,',%1.3f',q75_G);
fprintf(f,',%1.3f\n',q75_Ghat);
fprintf(f,',q90');
fprintf(f,',%1.3f',q90_G);
fprintf(f,',%1.3f\n',q90_Ghat);

fclose(f);












