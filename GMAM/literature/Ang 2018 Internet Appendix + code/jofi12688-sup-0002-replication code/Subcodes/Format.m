clear
load NewPreqin
load Factors

SP=[.03*ones(8,1);FactorM(1:118,7)];disc=ones(10,1);
for i=1:rows(SP);disc(i+1)=disc(i)/(1+SP(i));end
disc=disc(2:end);

d=CashFlows;
ID=unique(d(:,2));
d=sortrows(d,[2 15 20]);
disp('number of funds');rows(ID)
d(:,11)=d(:,11)/100000; % All cash flows are in million
d(:,12)=-99;
for typ=1:max(d(:,5))
    a=find(d(:,5)==typ & d(:,4)>=1992 & d(:,4)<2012);
    sta(typ,1:2)=[typ rows(unique(d(a,2)))];
end
for c=[2 3 5 9 16 18 20 22 29] % venture
    a=find(d(:,5)==c);
    d(a,12)=1;
end
for c=[1] % BO
    a=find(d(:,5)==c);
    d(a,12)=2;
end
for c=[4 6 30] % debt
    a=find(d(:,5)==c);
    d(a,12)=4;
end
    a=find(d(:,5)==7); % FoF
    d(a,12)=5;
for c=[8 12 32] % infra, nat res, timber
    a=find(d(:,5)==c);
    d(a,12)=6;
end    
    a=find(d(:,5)==11); % Real Estate
    d(a,12)=3;
% Kept special situation and Turnaround out
a=find(d(:,12)>0);d=d(a,:);
a=find(d(:,8)<=2);d=d(a,:); % EU, US

ID=unique(d(:,2));disp('number of funds');n=rows(ID)
c1=0;c2=0;inv=zeros(n,126);dis=zeros(n,126);DB=[];

for i=1:n
    a=find(d(:,2)==ID(i) & d(:,10)<3);
    da=d(a,:);na=rows(a);truevint=min(da(:,15));
  if truevint>=1984
    for j=1:na
        year=da(j,15);q=da(j,20);
        col=(year-1984)*4+q;
        if da(j,10)==1 % It is a contribution
            if da(j,11)<1
                inv(i,col)=inv(i,col)+da(j,11);
            else % if it is a high negative contribution, then it is a dividend
                dis(i,col)=dis(i,col)+da(j,11);c1=c1+1;
            end
        end
        if da(j,10)==2 % It is a distribution
            if da(j,11)>-1
                dis(i,col)=dis(i,col)+da(j,11);
            else
                inv(i,col)=inv(i,col)+da(j,11);c2=c2+1;
            end
        end
    end
    all=find(d(:,2)==ID(i));
    dall=d(all,:);
    if dall(end,10)==3
        NAV=dall(end,11);
    else
        NAV=0;
    end

    I=-sum(inv(i,:)'); % Fraction of Kcom called since all scaled
    D=sum(dis(i,:)');
    
    realized=D/(D+NAV);
    year=dall(end,15);q=dall(end,20);
    col=(year-1984)*4+q;
    dis(i,col)=dis(i,col)+dall(end,11);    
    D=sum(dis(i,:)');
    pvd=sum(dis(i,:)'.*disc);
    pvi=-sum(inv(i,:)'.*disc);
    %DB(i,:)=[da(1,2) truevint da(1,[8 7 12]) 0 I N D D/(D+N) Mult];
    DB(i,:)=[da(1,2) truevint da(1,[8 7 12]) 100*realized I NAV D/I pvd/pvi year q];
  end
end
a=find(isnan(DB(:,6))==1);DB(a,6)=100;
a=find(isnan(DB(:,4))==1);DB(a,4)=100;

%% Statistics for Table 1 - Panels A and B
a=find(DB(:,2)>=1984 & DB(:,2)<=2010 & DB(:,5)==1 & DB(:,3)==2); % US venture capital
%a=find(DB(:,2)>=1984 & DB(:,2)<=2010 & DB(:,5)==1); % venture capital
D=DB(a,:);
for year=1984:2010
    a=find(D(:,2)==year);
    stat(year-1983,:)=[rows(a) sum(D(a,6).*D(a,4))/sum(D(a,4)) sum(D(a,9).*D(a,4))/sum(D(a,4))];
end
stat(end+1,:)=mean(stat);
stat(end+1,:)=[rows(D) sum(D(:,6).*D(:,4))/sum(D(:,4)) sum(D(:,9).*D(:,4))/sum(D(:,4))];

statvc=stat;stat=[];

a=find(DB(:,2)>=1984 & DB(:,2)<=2010 & DB(:,5)==2 & DB(:,3)==2); % US Buyout
%a=find(DB(:,2)>=1984 & DB(:,2)<=2010 & DB(:,5)==2); % buyout
D=DB(a,:);
for year=1984:2010
    a=find(D(:,2)==year);
    stat(year-1983,:)=[rows(a) sum(D(a,6).*D(a,4))/sum(D(a,4)) sum(D(a,9).*D(a,4))/sum(D(a,4))];
end
stat(end+1,:)=mean(stat);
stat(end+1,:)=[rows(D) sum(D(:,6).*D(:,4))/sum(D(:,4)) sum(D(:,9).*D(:,4))/sum(D(:,4))];
statbo=stat;stat=[];

%% PMEs for post 1994 funds
a=find(DB(:,2)>=1994 & DB(:,2)<=2010 & DB(:,5)==2 & DB(:,3)==2); % US Buyout
D=DB(a,:);
for year=1994:2010
    a=find(D(:,2)==year);
    stat(year-1993,:)=[rows(a) mean(D(a,10)) median(D(a,10)) sum(D(a,10).*D(a,4))/sum(D(a,4))];
end
stat(end+1,:)=mean(stat);
stat(end+1,[1 4])=[rows(D) sum(D(:,10).*D(:,4))/sum(D(:,4))];

a=find(DB(:,2)>=1994 & DB(:,2)<=2010 & DB(:,5)==2 & DB(:,3)==1); % EU Buyout
D=DB(a,:);
for year=1994:2010
    a=find(D(:,2)==year);
    stat(year-1993,6:9)=[rows(a) mean(D(a,10)) median(D(a,10)) sum(D(a,10).*D(a,4))/sum(D(a,4))];
end
stat(end-1,:)=mean(stat);
stat(end,[6 9])=[rows(D) sum(D(:,10).*D(:,4))/sum(D(:,4))];

a=find(DB(:,2)>=1994 & DB(:,2)<=2010 & DB(:,5)==2); % Buyout
D=DB(a,:);stat=[];
for year=1994:2010
    a=find(D(:,2)==year);
    stat(year-1993,:)=[rows(a) mean(D(a,10)) median(D(a,10)) sum(D(a,10).*D(a,4))/sum(D(a,4)) sum(D(a,9).*D(a,4))/sum(D(a,4))];
end
statbo=stat;

a=find(DB(:,2)>=1994 & DB(:,2)<=2010 & DB(:,5)==1); % VC
D=DB(a,:);stat=[];
for year=1994:2010
    a=find(D(:,2)==year);
    stat(year-1993,:)=[rows(a) mean(D(a,10)) median(D(a,10)) sum(D(a,10).*D(a,4))/sum(D(a,4)) sum(D(a,9).*D(a,4))/sum(D(a,4))];
end
statvc=stat;

%% Keep only 1994-2010 funds
a=find(DB(:,2)>=1994 & DB(:,2)<=2010);
DB=DB(a,:)';
CashIn=-inv(a,41:end)';
CashOut=dis(a,41:end)';

save NewData DB CashIn CashOut    