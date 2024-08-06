%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw_bta
% Likelihood x Normal draw
% RW draw with accept/reject
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TrimPercent=1/T;
    
    Y=g_ret(5:end);  % Regress g_{t+1}-bta F_{t+1}-alpha on g_t-bta F_t and draw phi
    X=[ones(T-4,1) Factors(5:end,:)];
    
    indtemp = (Y<=quantile(Y,1-TrimPercent/2)) & ( Y>= quantile(Y,TrimPercent/2));% & (X<=quantile(X,1-TrimPercent/2)) & ( X>= quantile(X,TrimPercent/2));
    X=X(indtemp,:);
    Y=Y(indtemp);
    
    MU=[p_m_alpha;p_m_bta];
    LAM=diag([p_std_alpha^2;diag(p_std_bta).^2]); %;diag(p_std_bta).^2+diag(p_std_phi).^2]);
    
    LADLoop=3;
    E=eye(size(X,1));
    for k=1:LADLoop
        BTA=(X'*E*X)^-1*(X'*E*Y);
        E=diag(max(abs(Y-X*BTA),1e-10).^-1);
    end
    
    gp=cstd/4;
    BTA=(gp*BTA+MU)/(gp+1);
    std2=(X'*X)^-1*var(Y)*gp/(gp+1);
    
    BTAtmp=(BTA'+randn(1,Num_Factors+1)*chol(std2))';
    STDtmp=diag(std2).^.5;
    
    if [BTAtmp(2:Num_Factors+1)<max_bta ;BTAtmp(2:Num_Factors+1)>min_bta]
        bta=BTAtmp(2:Num_Factors+1);
    end


%%
alpha=mean(g_ret(5:end)-Factors(5:end,:)*bta);
f_ret=g_ret-Factors*bta-alpha;
if j>Burn
    iter=j-Burn;
    prev_x=E_f_ret;
    prev_x2=std_f_ret.^2+E_f_ret.^2;
    E_f_ret=1/iter*((iter-1)*prev_x+f_ret);
    
    tmp = 1/iter*( (iter-1)*prev_x2 + f_ret.^2 );                      % Update sum of squares
    std_f_ret = sqrt(max(tmp - E_f_ret.^2,0));
end

phi=0;

%% Draw sig_g
u=(g_ret(5:end)-Factors(5:end,:)*bta-alpha);
d1 =p_v5 + u'*u;
p_m5=d1/max_sig_g^2*set_sig_g;
nu1 = (T-4)+ p_m5;
P=rand*(invgamcdf(max_sig_g^2,nu1/2,d1/2)-invgamcdf(min_sig_g^2,nu1/2,d1/2))+invgamcdf(min_sig_g^2,nu1/2,d1/2);
if P==1
    sig_g=max_sig_g;
elseif P==0
    sig_g=min_sig_g;
else
    sig_g=invgaminv(P,nu1/2,d1/2)^.5;
end



