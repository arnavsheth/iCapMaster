% Keep only valid observations
for j=1:cols(svar)
    jj=colin(1,svar(1,j));
    a=find(d(:,jj)>-98);d=d(a,:);
end
a=find(d(:,ycol)>-98);d=d(a,:);

nd=rows(d);
st=zeros(100,1);
ws=1;

xx=d(:,colin);
x=xx(:,svar);
y=d(:,ycol);
cluster=d(:,clustercol);

cst=0;nli=cols(colin)+1;
store(2*nli:2*nli+5,spec)=-1;

for jf=1:cols(sfe)
    if sfe(jf)==1  % year
        col=4;cut=cuty;
        fes
        x=[x FE];
        store(2*nli,spec)=1;
    end
%     if sfe(jf)==2 % vintage
%         col=130;cut=cuty;
%         fes
%         x=[x FE];
%         store(2*nli+1,spec)=1;
%     end
    if sfe(jf)==2 % quarter
        col=23;cut=cuty;
        fes
        x=[x FE];
        store(2*nli+1,spec)=1;
    end   
    if sfe(jf)==3 % GP
        col=11;cut=cutgp;
        fes
        x=[x FE];
        store(2*nli+2,spec)=1;
    end
    if sfe(jf)==4 % Company
        col=289;cut=1;
        fes
        x=[x FE];
        store(2*nli+3,spec)=1;
    end
    if sfe(jf)==5 % Industry
        col=24;cut=cuty;
        fes
        x=[x FE];
        store(2*nli+4,spec)=1;
    end
    if sfe(jf)==6 % Industry
        col=26;cut=cuty;
        fes
        x=[x FE];
        store(2*nli+4,spec)=1;
    end
        if sfe(jf)==7 % Cross Indus year
        col=[24 4];cut=cuty;
        crosfes
        x=[x FE];
        store(2*nli+5,spec)=1;
    end
end

bb=hwhite(y,x);
%prt(bb)
a=clusterreg(y,x,cluster);
beta=a(:,1);tstat=a(:,3);
r2=bb.rbar;
nobs=bb.nobs;

%% Store result
for i=1:cols(svar)
    li=svar(1,i)*2-1;
    store(li,spec)=beta(i,1);
    store(li+1,spec)=tstat(i,1);
end

    store(2*nli+6,spec)=r2;
    store(2*nli+7,spec)=nobs;

spec=spec+1;