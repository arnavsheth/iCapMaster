% Code for estimation
% g_t = alpha + phi g_{t-1} + beta' F_t + sigma_g eps_t     Latent
% Observed CFs, unobserved returns
% Important: Data is created in Format.m
clear
load('C:\Users\lphalippou\Dropbox\PE_Bayesian\Code\NewData.mat')
% Add NAVs
CashOut=CashOut+NAV;
% select only vintage 1994-2008
a=find(DB(3,:)'>=1994 & DB(3,:)'<=2008);
DB=DB(:,a);CashOut=CashOut(:,a);CashIn=CashIn(:,a);
for i=1:size(CashOut,1)
    Ncashflows(i)=size(find(abs(CashOut(i,:)-CashIn(i,:))>0),2);
end
CashIn(94,:)=CashIn(94,:)+CashIn(95,:);CashIn=CashIn(9:94,:);
CashOut(94,:)=CashOut(94,:)+CashOut(95,:);CashOut=CashOut(9:94,:);

% Add factors
FactorM=xlsread('Factors','Sheet1');
FactBeg=(1994-1986)*4+2; % because consider cflows to be end of quarter
FactEnd=(2015-1986)*4+2; % The end is always June 2015
for h=[8 20 28 29 38 39 40]
FactorM(:,h)=FactorM(:,h)-FactorM(:,12); % Subtract Rf
end
for h=[5 6 46]
FactorM(:,h)=FactorM(:,h)-FactorM(:,7); % Subtract Vanguard
end
RF=FactorM(FactBeg:FactEnd,12);T=(2015-1994)*4+1;

wc=1;Burn = 500;Num_Sim =5000;

%% May want to do US only
%for typ=1:6 % 6=NatRes, 1=vc, 2=bo, 3=re, 4=credit, 5= FoF, 10=pool
for typ=[1:6 10]
    for wfactor=[1 3:5 8 9 10 11] % 1 = 1F, 3 = 3F, 4 = 4F, 5 = 1F, 6 = 3F, 7 = 4F, 8 = T1, 9 = T3, 10=T4, 11 = 5FF, 12 = Carhart + QMJ
    for er=1:10

rng(er)
typa % Gives the Iall set of funds
FactorModel

%% Remove outliers for estimation, but keep them for alpha computation
DBAll=DB(:,Iall);DivAll=CashOut(:,Iall);InvAll=CashIn(:,Iall);
p1=prctile(DBAll(10,:),5);p99=prctile(DBAll(10,:),95);Irestrict=find(DBAll(10,:)>=p1 & DBAll(10,:)<=p99);
div=DivAll(:,Irestrict);            % dividend
Inv=InvAll(:,Irestrict);            % investment
CF_mat=div-Inv;
DBin=DBAll(:,Irestrict);
Num_Funds=size(div,2)

Parameter

% Start burning
for j = 1:Burn
Draw_g_ret
Draw_bta
  % Update parameter matrix
   % A and B are Gamma distribution parameters, and sig is the implied standard error of the gamma distribution from A and B
  thistheta = [alpha phi bta' sig_g];               
  theta_mat = [theta_mat; thistheta];
end   % Number of simulations

% Start actual ones
theta_mat=[];PME_arr=[];

for j=Burn+1:Burn+Num_Sim
  Draw_g_ret
  Draw_bta
 % Update parameter matrix
  r2=var((Factors(2:end,:)-phi*Factors(1:end-1,:))*bta)/var(g_ret(2:end)-phi*g_ret(1:end-1));
  thistheta = [alpha phi bta' sig_g];
  theta_mat = [theta_mat; thistheta];
end
    disp('     alpha    phi    bta1    bta2     bta3    sig_g   A   B   sig');
    printmatrix(mean(theta_mat),strvcat('Mean'));
    printmatrix(std(theta_mat),strvcat('Std'));
    kip=mean(theta_mat);
    Beta(wc,1:cols(kip))=kip;
    output
    wc=wc+1;
end    
end
end
%save Full


