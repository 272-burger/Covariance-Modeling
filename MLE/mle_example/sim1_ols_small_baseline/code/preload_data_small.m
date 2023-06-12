% This script generates the clustering structure and location variables we
% need for inference in simulation. Results are saved in 'clustering.mat'. 

% addpath('../data');

% coordinates of locations
eldata = importdata('../data/ElectionViolenceTABLE2.csv');
XCoord = eldata.data(:,9);
YCoord = eldata.data(:,10);

% % generate a map 4 times as large 
% XCoord = [XCoord; 2*ceil(max(XCoord))-XCoord];
% YCoord = [YCoord;YCoord];
% XCoord = [XCoord;XCoord];
% YCoord = [YCoord;2*floor(min(YCoord))-YCoord];

% plot(XCoord,YCoord,'o')

ind = 1:2:length(XCoord);
coord = [XCoord YCoord];
coord_unique = coord(ind,:);
N = length(coord_unique);
T = 2;


%% clustering

G_max = ceil((N*T)^(1/3));
G_vec = 2:G_max;

% cluster by location 
group_location = kron((1:N)',[1;1]);

% K-medoids
group_matrix_km = zeros(N,length(G_vec));
for i = 1 : length(G_vec)
    group_matrix_km(:,i) = kmedoids(coord_unique,G_vec(i),...
        'Replicates',100);
end
group_matrix_km = kron(group_matrix_km,[1;1]);

% % Penalized Hierarchical Clustering
% Ng_lb = 20;
% group_matrix_hc = penalized_hierarchical_clustering(squareform...
%     (pdist(coord_unique)),G_vec,Ng_lb);
% group_matrix_hc = kron(group_matrix_hc,[1;1]);

% tabulate(group_matrix_hc(:,1))

% % K-medoids - G_bar = 10
% G_vec = 2:10;
% group_matrix_km_G10 = zeros(N,length(G_vec));
% for i = 1 : length(G_vec)
%     group_matrix_km_G10(:,i) = kmedoids(coord_unique,G_vec(i),...
%         'Replicates',100);
% end
% group_matrix_km_G10 = kron(group_matrix_km_G10,[1;1]);


%% save data

dis_mat = squareform(pdist(coord));
dis_mat_unique = squareform(pdist(coord_unique));

save('../temp/preload_data_small.mat', 'group_location',... 
    'group_matrix_km',...
    'dis_mat','dis_mat_unique','coord');


