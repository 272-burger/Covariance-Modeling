function [F_sim_cv p_value_sim]  = simulation(loc, K1, K2, q, nsim, alpha, F_value)

          xloc = loc(:,1);
          yloc = loc(:,2);
          N = length(xloc); % N is the sample size
          nx = max(xloc)-min(xloc);
          ny = max(yloc)-min(yloc);
   
          K = K1*K2+K1+K2;
          
          basis = complex(zeros(N,K+1));
          count = 1;
          
          for k1 = 0:1:K1
                   for k2 = 0:1:K2
                          tempest = exp(-1i*2*pi*(k1*xloc/nx + k2*yloc/ny));
                          basis(:,count) = tempest - mean(tempest) ; % Demeaned Basis
                          count = count+1;
                   end
          end;
          
          basis(:,1) = [];
          
          F_sim = zeros(nsim,1);  %
          
          for rep_sim = 1 : nsim;    
              
               epsj = randn(N,q);
               eta = sum(epsj)'/sqrt(N);
               
               sum_xi = zeros(q,q);
               
               for k = 1:1:K 
                temp = basis(:,k)'*epsj/sqrt(N);
                sum_xi = sum_xi+real(temp'*temp);
               end
               sum_xi = sum_xi/K;

               F_sim(rep_sim,1) = eta'*(sum_xi\eta) / q;

          end 
          
            F_sim = sort(F_sim);
            
            F_sim_cv = F_sim((nsim+1)*(1-alpha),1);
            
            temp = F_value * ones(nsim,1) < F_sim;
             
            p_value_sim = sum(temp)/(nsim+1);



