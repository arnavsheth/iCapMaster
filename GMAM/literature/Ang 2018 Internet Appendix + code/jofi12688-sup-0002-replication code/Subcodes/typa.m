% scol is the column for the alternative AP model to be used in each
% category
Beta=zeros(10,20);isfactor=1;
cstd=100;
conf=1;
Iall=[];
if conf==2;p_m_bta=[2 0 0 0];end
c=find(DB(5,:)==31);DB(5,c)=0;

if typ==1 || typ==10  % vc
    scol=29;
    Iall=[Iall find(DB(5,:)==1)];
    p_m_bta=[1.70 0.80 -0.85 0.5 0];
end

if typ==2 || typ==10  % bo
    scol=8;
    Iall=[Iall find(DB(5,:)==2)];
    p_m_bta=[1.25 0.1 0.6 0.7 0];
    
end

if typ==3 || typ==10  % re
    scol=38;
    Iall=[Iall find(DB(5,:)==3)];
    p_m_bta=[0.6 0.4 0.6 0.5 0];
    
end

if typ==4 || typ==10 % credit
    scol=20;
    Iall=[Iall find(DB(5,:)==4)];
    p_m_bta=[0.7 1.3 1.45 0.5 0];
    
end

if typ==5 % FoF
    scol=8;
    Iall=[Iall find(DB(5,:)==5)];
    p_m_bta=[1 1.3 1.45 0.5 0];
    
end

if typ==6 || typ==10 % Infra
    scol=8;
    Iall=[Iall find(DB(5,:)==6)];
    p_m_bta=[0.6 1.3 1.45 0.5 0];
end

if typ==10 % All
    scol=8;
    p_m_bta=[1.6 0.55 0.05 0.5 0];
end

if conf==3;p_m_bta=p_m_bta-0.5;end
if conf==4;p_m_bta=p_m_bta+0.5;end
p_m_alpha=0;