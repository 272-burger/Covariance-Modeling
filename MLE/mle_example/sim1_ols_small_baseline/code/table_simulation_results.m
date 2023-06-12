

%% summary table - Bias, RMSE, Size (Median bias, MAD in IV cases)
texOutput = '../output/table_simulation_results.csv';
if exist(texOutput,'file')
    delete(texOutput);
end
f = fopen(texOutput,'a');
if strcmp(modelName,'ols')
    fprintf(f,'Method,Bias, RMSE,Size\n');
elseif strcmp(modelName,'iv')
    fprintf(f,'Method,Median Bias, MAD,Size\n');
end

% SK
fprintf(f,'SK');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,size_sk);
fprintf(f,'\n');

% UNIT-U
fprintf(f,'UNIT-U');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,size_loc_u);
fprintf(f,'\n');

% UNIT
fprintf(f,'UNIT');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,size_loc);
fprintf(f,'\n');

% CCE
fprintf(f,'CCE');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_loc,rmse_loc,sizeGstarB_cce);
fprintf(f,'\n');

% IM
fprintf(f,'IM');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_im,rmse_im,sizeGstar_im);
fprintf(f,'\n');

% CRS
fprintf(f,'CRS');
fprintf(f,' , %1.3f , %1.3f , %1.3f ',mean_crs,rmse_crs,sizeGstar_crs);
fprintf(f,'\n');

fclose(f);












