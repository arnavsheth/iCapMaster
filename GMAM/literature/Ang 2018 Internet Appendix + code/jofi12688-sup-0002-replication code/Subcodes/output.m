nl=2+Num_Factors;
btamean=mean(theta_mat(1:end,3:nl),1); % Mean theta: [alpha phi bta' sig_g A B sig]
Amean=mean(theta_mat(:,1:2),1);

%% Look at alpha that clears PME=1
[alpha_full,eval]=fminsearch(@(x)MPME_full(x,btamean,Factors,RF,E_f_ret,DivAll,InvAll,ones(size(DivAll,2),1)),0)
deltaF=Amean(1)-alpha_full;
F=RF+E_g_ret-deltaF;
G=F;

disc=ones(T+1,1);for ci=2:T+1;disc(ci,1)=disc(ci-1,1)/(1+F(ci-1,:));end
P=zeros(size(DivAll,2),1);
for ci=1:size(DivAll,2);
    P(ci,1)=sum(DivAll(:,ci).*disc)/sum(InvAll(:,ci).*disc); % PME of each fund use F
end

disp('PME for full sample (mean log, mean, median)')
disp('mean PME should be one by construction')
[mean((P)) median((P))]

%% Store results
storg(1:T,wc)=F; % Adjusted return series (Rf+g_t)
storf(1:T,wc)=E_f_ret;

storg_std(1:T,wc)=std_g_ret;
storf_std(1:T,wc)=std_f_ret;
store(wc,1:nl)=[(1+alpha_full)^4-1 (1+Amean(1))^4-1 btamean];

store(wc,22:25)=[mean((P)) median((P)) LL_new Num_Funds];
store(wc,26:27)=Amean;
a=[Num_Factors std(theta_mat(:,1:2)) 0 std(theta_mat(:,3:nl))];
store(wc,10:size(a,2)+9)=a; % Column 10 onwards (col 12 gets persistence std)
