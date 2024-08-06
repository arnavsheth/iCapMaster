clear
% This is what run on OSC, quick version
wi=1;Burn = 200;Num_Sim = 4000;
ro=0;T=80;Num_Factors=1;isfactor=1;RF=zeros(T,1);   
p_m_alpha=0.01;storb=[];storc=[];
nfund_base=127;
volall=.1;
HoldingP=ceil(28*rand(50000,1));
in_money=0;
mngtfees=0;

for nround=1:10 % number of times the simulation is run
            
    rng('shuffle')
    % default return series
    mkt=.02+.1*randn(80,1);
    vvrnd=randn(80,1);
    f=filter(1,[1,-ro],volall*vvrnd);
    Factors=mkt;
    tret=.01+f+Factors*1.5;

    for ivol=[.005 .05 .25 .5]; %.25
    dRetRnd=min(ivol*randn(50000,80),.5);
    Ret=1+[zeros(50000,1) max(repmat(tret',50000,1)+dRetRnd,-.95)];
    CumRet=cumprod(Ret,2);
    DGP_ret=mean(Ret(:,2:end)-1)';
    mean(DGP_ret);
    w=1;

%% Gross of fees Value (and Inv) Matrix
    Invt=repmat(eye(20),nfund_base,1);
    Von=Invt;
    for i=1:nfund_base
        for h=1:20
            stop=min(h+HoldingP(w),80);
            Von(w,h+1:stop)=-1;
            w=w+1;
        end
    end
  % New every quarter
  for q=2:T-19
      nfund_next=3+floor(sqrt(q));
        Invt=[Invt zeros(size(Invt,1),1);zeros(20*nfund_next,q-1) repmat(eye(20),nfund_next,1)];
        for i=1:nfund_next
        for h=q:q+19
            stop=min(h+HoldingP(w),81);
            Von(w,h)=1;
            Von(w,h+1:stop)=-1;
            w=w+1;
        end
        end      
  end
Invt(:,81)=0;        
sum(sum(Invt));        

    for carry_rate=[.1 .2 .3];
    p_m_bta=1.5-2*carry_rate;
    Div=Invt*0;CarryProv=Div;ChangeInCarryProv=Div;
    nf=1;Invest=[];Divest=[];


        %% Pool investments to form funds and subtract fees
        for i=1:20:size(Invt,1)-1
            totfee=0;
            for q=2:81
                a=find(Von(i:i+19,q)==-1);
                R=CumRet(i+a-1,q)./CumRet(i+a-1,q-1);
                Von(i+a-1,q)=Von(i+a-1,q-1).*R;
            end
            for li=0:19
                a=find(Von(i+li,:)>0);
                for h=a(1):a(end)
                CarryProv(i+li,h)=carry_rate*blsprice(Von(i+li,h),ivol,0,a(end)-h,1);
                end
                ChangeInCarryProv(i+li,a(1):a(end))=[CarryProv(i+li,a(1)) CarryProv(i+li,a(2):a(end))-CarryProv(i+li,a(1):a(end)-1)];
                Div(i+li,a(end))=Von(i+li,a(end));
            end
                Invest(:,nf)=sum(Invt(i:i+19,:))';
                Divest(:,nf)=sum(Div(i:i+19,:))';
                nf=nf+1;
        end
Vona=Von-ChangeInCarryProv;
sum(sum(ChangeInCarryProv))
for q=2:T+1        
        a=find(Vona(:,q)>0 & Vona(:,q-1)>0);
        TrueRet(q-1)=sum(Vona(a,q))./sum(Vona(a,q-1))-1;
end             
a=ols(TrueRet',[ones(80,1) Factors]);truebeta=a.beta(2);         
        %% Sub-code here: Specify priors and the like
        div=Divest;Inv=Invest;Num_Funds=nf-1
        DivAll=Divest;InvAll=Invest;
        CF_mat=div-Inv;
        %AnyparamQtryWrongError
        Parameter %%%%%%%%%%%%%
        p_m_phi=0;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Simulation Loop
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('starting with burning')
        for j = 1:Burn
            Draw_g_ret
            Draw_bta
        end
        disp('done with burning')
        
        theta_mat=[];PME_arr=[];
        for j = Burn:Burn+Num_Sim
            Draw_g_ret
            Draw_bta
            r2=var((Factors(2:end,:)-phi*Factors(1:end-1,:))*bta)/var(g_ret(2:end)-phi*g_ret(1:end-1));
            thistheta = [alpha bta' phi sig_g alpha sig];
            theta_mat = [theta_mat; thistheta];
%             if abs(j/2000-round(j/2000))<0.0001
%                 bb=[trueret' E_g_ret];cc=corrcoef(bb)
%             end
        end
        
        bb=[E_g_ret TrueRet'];cc=corrcoef(bb);
        x=[ones(80,1) Factors];ab=ols(bb(:,1),x);
        storb=[storb bb];
        storc=[storc;cc(1,2) truebeta mean(theta_mat) std(bb)];
        %save FeeOption
    end
    end
end
%%
N=12;
for i=1:N
    sa(i,:)=mean(storc(i:N:end,:));
end
save FeeOption





