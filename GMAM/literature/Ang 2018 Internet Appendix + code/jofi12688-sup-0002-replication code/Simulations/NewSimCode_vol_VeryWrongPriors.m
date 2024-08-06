clear
% This is what run on OSC, quick version
wi=1;Burn = 10;Num_Sim = 90;
ro=0;T=80;Num_Factors=1;isfactor=1;RF=zeros(T,1);   
p_m_alpha=0;p_m_bta=.5;storb=[];storc=[];
nfund_base=127;
volall=.05;
HoldingP=ceil(28*rand(50000,1));
in_money=0;
mngtfees=0;
carry_rate=0;

for nround=1:10 % number of times the simulation is run
            
    rng('shuffle')
    % default return series
    mkt=.02+.1*randn(80,1);
    vvrnd=randn(80,1);
    f=filter(1,[1,-ro],volall*vvrnd);
    Factors=mkt;
    tret=.01+f+Factors*1.5;

    for ivol=[.005 .05 .125 .25 .375 .5 .625 .75]; %.25
    dRetRnd=min(ivol*randn(50000,80),.5);
    Ret=1+[zeros(50000,1) max(repmat(tret',50000,1)+dRetRnd,-.95)];
    CumRet=cumprod(Ret,2);
    DGP_ret=mean(Ret(:,2:end)-1)';
    mean(DGP_ret)
    w=1;

%% Gross of fees Value (and Inv) Matrix
    Inv=repmat(eye(20),nfund_base,1);
    Von=Inv;
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
        Inv=[Inv zeros(size(Inv,1),1);zeros(20*nfund_next,q-1) repmat(eye(20),nfund_next,1)];
        for i=1:nfund_next
        for h=q:q+19
            stop=min(h+HoldingP(w),81);
            Von(w,h)=1;
            Von(w,h+1:stop)=-1;
            w=w+1;
        end
        end      
  end
Inv(:,81)=0;        
sum(sum(Inv))        
Div=Inv*0;

        %% Pool investments to form funds and subtract fees
        nf=1;Invest=[];Divest=[];
        for i=1:20:size(Inv,1)-1
            totfee=0;
            for q=2:81
                a=find(Von(i:i+19,q)==-1);
                R=CumRet(i+a-1,q)./CumRet(i+a-1,q-1);
                n=rows(a);
                if n>0 & totfee<200*mngtfees
                    fee=(20*mngtfees)/4/n;
                else
                    fee=0;
                end 
                Von(i+a-1,q)=Von(i+a-1,q-1).*R-fee;
                Inv(i+a-1,q)=Inv(i+a-1,q)+fee;
                totfee=totfee+fee;
            end
            for li=0:19
                a=find(Von(i+li,:)>0);
                Div(i+li,a(end))=Von(i+li,a(end));
            end
  
                TotInv=sum(sum(Inv(i:i+19,:)));
                dd=sum(Div(i:i+19,:));
                cdd=cumsum(dd);carrypaid_tot=0;
            if cdd(end)>TotInv*(1.08^3.5)
                in_money=in_money+1;
                carry=carry_rate*(cdd-TotInv);
                dcarry=[0 carry(2:end)-carry(1:end-1)];
                ad=find(dcarry>0 & carry>0);
                for cc=ad
                    a=find(Div(i:i+19,cc)>0);n=rows(a);
                    duecarry=carry(cc);
                    fcp=(duecarry-carrypaid_tot)/sum(Div(i:i+19,cc));
                    olddiv=Div(i:i+19,cc);
                    Div(i:i+19,cc)=olddiv*(1-fcp);
                    Von(i:i+19,cc)=Von(i:i+19,cc)-olddiv+Div(i:i+19,cc);
                    carrypaid_tot=carrypaid_tot+sum(fcp*olddiv);
                end
            end
                feetrack(nf,1)=carrypaid_tot;
                feetrack(nf,2)=totfee;
                Invest(:,nf)=sum(Inv(i:i+19,:))';
                Divest(:,nf)=sum(Div(i:i+19,:))';
                nf=nf+1;
        end

for q=2:T+1        
        a=find(Von(:,q)>0 & Von(:,q-1)>0);
        TrueRet(q-1)=sum(Von(a,q))./sum(Von(a,q-1))-1;
end             
        
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
            thistheta = [alpha bta' phi sig_g alpha B sig r2];
            theta_mat = [theta_mat; thistheta];
%             if abs(j/2000-round(j/2000))<0.0001
%                 bb=[trueret' E_g_ret];cc=corrcoef(bb)
%             end
        end
        
        bb=[E_g_ret DGP_ret TrueRet'];
        error=bb(:,1)-bb(:,3);
        cc=corrcoef(bb)
        mean(bb)
        x=[ones(80,1) bb(:,2)];ab=ols(bb(:,1),x);prt(ab)
        storb=[storb bb];
        storc=[storc;cc(1,2) mean(error.^2)];

    end        
end
%%
for i=1:8
    sa(i,1:2)=mean(storc(i:8:end,:));
    sa(i,3:4)=median(storc(i:8:end,:));
end






