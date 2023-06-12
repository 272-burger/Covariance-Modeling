function [p_value_star] = main_web_iv(ydep,x_reg,z_mat,Xcoord,Ycoord,R,r)
%% This is the main file to implement the "asymptotic F test" and "series 
%% nonstandard test" for spatial data developed in Sun, Y. and M.S. Kim (2012), 
%% "Asymptotic F Test in a GMM Framework with Cross Sectional Dependence."  

q = size(R,1);
N = size(ydep,1);

%%  Standardized locations
[xloc, yloc] = form_regular_lattice(Xcoord,Ycoord);
loc = [xloc yloc];

K_rule = .4*(N/2)^(2/3);
K1 = round(sqrt(K_rule+1)-1);
K2 = K1;

K = (K1+1)*(K2+1)-1;

%% Asymptotic F test 
[F_new,~] = F_star_iv(R,r,ydep,x_reg,z_mat,loc,K1,K2);
% F_star_cv = finv(1-alpha,q,2*K-q+1);
p_value_star = 1-fcdf(F_new,q,2*K-q+1);
% test = (p_value_star<alpha);
