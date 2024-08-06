clear
load Full
load('C:\Users\lphalippou\Dropbox\PE_Bayesian\Code\NewData.mat')
TS=xlsread('C:\Users\lphalippou\Dropbox\PE_Bayesian\Indices.xlsx');

% Add NAVs
CashOut=CashOut+NAV;
% select only vintage 1994-2008
a=find(DB(3,:)'>=1994 & DB(3,:)'<=2008);
DB=DB(:,a);CashOut=CashOut(:,a);CashIn=CashIn(:,a);
CashIn(94,:)=CashIn(94,:)+CashIn(95,:);CashIn=CashIn(9:94,:);
CashOut(94,:)=CashOut(94,:)+CashOut(95,:);CashOut=CashOut(9:94,:);
FactorM=xlsread('Factors','Sheet1');
FactBeg=(1994-1986)*4+2; % because consider cflows to be end of quarter
FactEnd=(2015-1986)*4+2; % The end is always June 2015
T=(2015-1994)*4+1;F=FactorM(FactBeg:FactEnd,7);
a=find(isnan(DB(4,:))==1);DB(4,a)=100;
%% PME each fund on line 12
    disc=ones(T+1,1);for ci=2:T+1;disc(ci,1)=disc(ci-1,1)/(1+F(ci-1,:));end
    for ci=1:size(CashOut,2);
    DB(12,ci)=sum(CashOut(:,ci).*disc)/sum(CashIn(:,ci).*disc); % PME of each fund use F
    end
    
%% Table 2    
    for year=1994:2008
        Iall=find(DB(5,:)==2 & DB(3,:)==year);
        tabl(year-1993,:)=[size(Iall,2) median(DB(12,Iall)') sum(DB(12,Iall).*DB(4,Iall))/sum(DB(4,Iall))];
        Iall=find(DB(5,:)==1 & DB(3,:)==year);
        tabl(year-1970,:)=[size(Iall,2) median(DB(12,Iall)') sum(DB(12,Iall).*DB(4,Iall))/sum(DB(4,Iall))];
    end

%% Table 3
% Select model with highest likelihood
c=1;w=1;G=[];
for h=1:10:rows(store)
    [a b]=max(store(h:h+9,24));
    Tabl(c,1:7)=store(h+b-1,[3:7 1 24]);
    Tabl(c+1,1:6)=store(h+b-1,[14:18 11]);
    G(:,w)=[storg(:,h+b-1);store(h+b-1,24)];
    c=c+2;w=w+1;
end
w=1;
for h=1:8:cols(G)
    [a b]=max(G(end,h:h+7)');
    GG(:,w)=G(1:end-1,h+b-1);
    w=w+1;
end
% GG is VC, BO, RE, credit, FoF, NatRes, All
% keep all VC, BO, RE and All
GG=GG(9:84,[1 2 3]); % Q1 96 to Q4 2014
Table4(:,1)=4*mean(GG)';
Table4(:,2)=2*std(GG)';
Table4(:,3:4)=4*prctile(GG,[25 75])';
for i=1:3
Table4(i,5)=sacf(GG(:,i),1);
end
aa=accumulate(log(1+GG));plot(aa)

%% Table 5
TS=TS(1:76,:);
y=TS(:,2);
xx=[ones(76,1) TS(:,12:19)];
for c=2:cols(xx)
    b=find(isnan(xx(:,c))==0);be=max(b);
    a=nwest(y(1:be),xx(1:be,1:c),4);
    Table5(1:2:2*c,c-1)=a.beta;
    Table5(2:2:2*c,c-1)=a.tstat;
    Table5(19,c-1)=a.rsqr;
    Table5(20,c-1)=a.nobs;
end

aa=accumulate(log(1+TS(:,[2 9 11])));plot(aa)