
README File

This document summarizes the Matlab codes available to implement "asymptotic F test" and 
"series nonstandard test" for spatial data developed in Sun, Y. and M.S. Kim (2012), 
"Asymptotic F Test in a GMM Framework with Cross Sectional Dependence." 

To run the programs, you need the following files:
(1) main_web.m (2) F_series.m (3) F_star.m (4) fminsearchbnd.m (5) form_regular_lattice.m
(6) GS.m (7) ipdm.m (8) optimalK.m (9) parseargs.m (10) simulation.m (11) variogram.m
(12) variogramfit.m. 

The main file is "main_web." Open this file and define
(1) output file name
(2) dependent variable data as "y_dep"
(3) explanatory variable data as "x_reg"
(4) two dimensional location data as "Xcoord" and "Ycoord"
(5) restrictions in your joint null hypotheses, ie. "R" and "r" in H0: R*theta = r. 

The dimension of each variable should be as follows:
y_dep, Xcoord, Ycoord : N by 1
x_reg : N by d
R : q by d
r : q by 1
where N is # of observations, d is # of regressors, and q is # of restrictions.

After defining the variables above, run "main_web," and you will have the test resuls. 

  
