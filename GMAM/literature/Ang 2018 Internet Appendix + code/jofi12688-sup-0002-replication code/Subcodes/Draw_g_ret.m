%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Draw_g_ret and Drar_sig are combined together
% Single state updating
% g_t = alpha+phi*g_(t-1)+bta (RM(t)-phi*RM(t-1))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%phi=0;

g_ret_old = g_ret;                                                   % Store old f_ret in case of no update
accept_vec = zeros(T,1);
stepLength=0.05*ones(T,1);
sig_g = max_sig_g/2; % could be set to min

for t = randperm(T) 
  temp_mu=alpha+Factors(t,:)*bta;
  org_std = sig_g; 
  
  prop_std =  stepLength(t);
  qg = temp_mu + prop_std *randn(1,1);
  g_ret_new=g_ret;
  g_ret_new(t)=qg;
  
  % adjusted for using prop_std instead of org_std to propose
    LL_Adjust = -.5*(qg-temp_mu)^2/(org_std^2) + .5*(qg-temp_mu)^2/(prop_std^2)+ 0.5 *(g_ret(t)-temp_mu)^2/(org_std^2) - 0.5*(g_ret(t)- temp_mu)^2/(prop_std^2); 
  
  % Log Likelihoods
     [LL_old,PME_old] = LL_allfunds_Norm(div,Inv,g_ret+RF,max_sig,mpme);
     [LL_new,PME_new] = LL_allfunds_Norm(div,Inv,g_ret_new+RF,max_sig,mpme);
      
  % Accept/reject
  Likratio = exp(LL_new - LL_old + LL_Adjust);
  accept_prob = min(Likratio,1);                                     

  accept = 0;
  if rand(1,1) < accept_prob && qg < max_ret && qg > min_ret && abs(mean(PME_new)-1)<max(delta,abs(mean(PME_old)-1))           % Accept with prob accept_prob
    accept = 1;
    g_ret(t) = qg;
  end
  
  % Acceptance
  accept_vec(t) = accept;

end   % Loop for t

%%
if j > Burn
  acceptg_ret = 1/(j-Burn)*(acceptg_ret*(j-Burn-1) + accept_vec);
end
if j > Burn && rem(j,Burn) == 0
   stepLength=min(max(std_g_ret,0.02),.5);    
 end

    [LL,PME,sig,mpme] = LL_allfunds_Norm(div,Inv,g_ret+RF,max_sig,mpme);

PME_arr=[PME_arr PME];
mpme_arr=[mpme_arr mpme];

% Store g_ret
if j <= Burn

  iter = j;                                                     % Number of iterations                       
  prev_x = E_g_ret;                                                  % Previous mean and sum of squres
  prev_x2 = std_g_ret.^2 + E_g_ret.^2;
  
  E_g_ret = 1/iter*((iter-1)*prev_x + g_ret);                        % Update mean
   
  tmp = 1/iter*( (iter-1)*prev_x2 + g_ret.^2 );                      % Update sum of squares
  std_g_ret = sqrt(max(tmp - E_g_ret.^2,0));

end 

if j > Burn

  iter = j-Burn;                                                     % Number of iterations                       
  prev_x = E_g_ret;                                                  % Previous mean and sum of squres
  prev_x2 = std_g_ret.^2 + E_g_ret.^2;
  
  E_g_ret = 1/iter*((iter-1)*prev_x + g_ret);                        % Update mean
   
  tmp = 1/iter*( (iter-1)*prev_x2 + g_ret.^2 );                      % Update sum of squares
  std_g_ret = sqrt(max(tmp - E_g_ret.^2,0));

end   % Store g_ret