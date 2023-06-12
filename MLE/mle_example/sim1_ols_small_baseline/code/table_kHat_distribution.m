

%% distribution of selected number of groups (kHat)
texOutput = '../output/table_kHat_distribution.csv';
if exist(texOutput,'file')
    delete(texOutput);
end
f = fopen(texOutput,'a');
fprintf(f,'   , kHat=2');
fprintf(f,' , %d',3:G_bar);
fprintf(f,'\n');

% CCE
fprintf(f,'CCE  ');
fprintf(f,' , %1.3f',GdistB_cce);
fprintf(f,'\n');

% IM 
fprintf(f,'IM');
fprintf(f,' , %1.3f',Gdist_im);
fprintf(f,'\n');

% CRS 
fprintf(f,'CRS');
fprintf(f,' , %1.3f',Gdist_crs);
fprintf(f,'\n');

fclose(f);












