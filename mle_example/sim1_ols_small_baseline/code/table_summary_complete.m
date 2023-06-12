

%% complete summary table
% Bias, RMSE, Size, Power at [-1,-.5,.5,1] (Median Bias & MAD in IV cases)
texOutput = '../output/table_summary_complete.csv';
if exist(texOutput,'file')
    delete(texOutput);
end
f = fopen(texOutput,'a');
if strcmp(modelName,'ols')
    fprintf(f,'Method,Mean, RMSE,Size,Power\n');
elseif strcmp(modelName,'iv')
    fprintf(f,'Method,Median, MAD,Size,Power\n');
end
fprintf(f,',,,,-1,-0.5,0.5,1\n');

% SK
fprintf(f,'SK');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,size_sk);
fprintf(f,' , %1.3f',powerGrid_sk);
fprintf(f,'\n');

% UNIT-U
fprintf(f,'UNIT-U');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,size_loc_u);
fprintf(f,' , %1.3f',powerGrid_loc_u);
fprintf(f,'\n');

% UNIT
fprintf(f,'UNIT');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,size_loc);
fprintf(f,' , %1.3f',powerGrid_loc);
fprintf(f,'\n');

% CCE
fprintf(f,'CCE');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,sizeGstarB_cce);
fprintf(f,' , %1.3f',powerGrid_cce);
fprintf(f,'\n');

% IM
fprintf(f,'IM');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_im,rmse_im,sizeGstar_im);
fprintf(f,' , %1.3f',powerGrid_im);
fprintf(f,'\n');

% CRS
fprintf(f,'CRS');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_crs,rmse_crs,sizeGstar_crs);
fprintf(f,' , %1.3f',powerGrid_crs);
fprintf(f,'\n');
fclose(f);












