
%% quantile of adjusted p-value threshold (alphaHat)
texOutput = '../output/table_alphaHat_distribution.csv';
if exist(texOutput,'file')
    delete(texOutput);
end
f = fopen(texOutput,'a');
fprintf(f,'   , Quantile\n');
fprintf(f,',0.1,0.25,0.5,0.75,0.9\n');
% fprintf(f,'\n');

% CCE
if_oversize = cvalGstarB>sum((repmat(G_list,B,...
    1) == repmat(GstarB,1,...
    length(G_list))).*repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1),2);
pValTH = 2*(1-tcdf(cvalGstarB,GstarB-1));
pValTH = if_oversize.*pValTH+(1-if_oversize).*sigLevel;
pValTH_q_cce = quantile(pValTH,[.1,.25,.5,.75,.9]);

fprintf(f,'CCE  ');
fprintf(f,' , %1.3f',pValTH_q_cce);
fprintf(f,'\n');

% IM
if_oversize = cvalGstar_im>sum((repmat(G_list,B,...
    1) == repmat(Gstar_im,1,...
    length(G_list))).*repmat(tinv(1-sigLevel/2,...
    max(group_matrix_km)-1),B,1),2);
pValTH = 2*(1-tcdf(cvalGstar_im,Gstar_im-1));
pValTH = if_oversize.*pValTH+(1-if_oversize).*sigLevel;
pValTH_q_im = quantile(pValTH,[.1,.25,.5,.75,.9]);

fprintf(f,'IM');
fprintf(f,' , %1.3f',pValTH_q_im);
fprintf(f,'\n');

% CRS
pValTH_q_crs = quantile(pvalAdjGstar_crs,[.1,.25,.5,.75,.9]);

fprintf(f,'CRS');
fprintf(f,' , %1.3f',pValTH_q_crs);
fprintf(f,'\n');

fclose(f);


