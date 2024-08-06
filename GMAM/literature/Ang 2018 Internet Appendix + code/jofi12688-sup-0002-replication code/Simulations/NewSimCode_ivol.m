clear
% Wrong error structure
wi=1;Burn = 200;Num_Sim = 5000;
ro=0;T=80;Num_Factors=1;isfactor=1;RF=zeros(T,1);   
p_m_alpha=0;storb=[];storc=[];
nfund_base=127;
HoldingP=ceil(28*rand(50000,1));
in_money=0;
mngtfees=0;
carry_rate=0;

for nround=1:20 % number of times the simulation is run
            
    rng('shuffle')
    % default return series
    mkt=.02+.1*randn(80,1);
    vvrnd=randn(80,1);
    
        wprior=0;volall=.1;
        f=filter(1,[1,-ro],volall*vvrnd);
        Factors=mkt;
        truebeta=1.5;
        tret=.01+f+Factors*truebeta;
        p_m_bta=truebeta-wprior;
        
    for ivol=[.005 .05 .1 .2 .35 .5 .75]

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
                n=size(a,1);
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
thistheta = [alpha bta' phi sig_g sig];
theta_mat = [theta_mat; thistheta];
        end
        
        bb=[E_g_ret TrueRet' E_f_ret f];
        error=bb(:,1)-bb(:,2);
        x=[ones(80,1) Factors];ab=ols(bb(:,1),x);
        cc=corrcoef(bb);
        storb=[storb bb];
        storc=[storc;cc(1,2) ab.beta(2) mean(theta_mat) std(bb) wprior volall truebeta ivol];
        nround

    end
end
%%
N=7;sat=[];Fanal=[];sa=[];
for i=1:N
    sa(i,:)=mean(storc(i:N:end,:));
    sat(i,1)=mean(storc(i:N:end,4)-storc(i:N:end,2));
    sat(i,2)=std(storc(i:N:end,3)-storc(i:N:end,2));
    sat(i,3)=prctile((storc(i:N:end,4)-storc(i:N:end,2)),75)-prctile((storc(i:N:end,4)-storc(i:N:end,2)),25);
    sat(i,5)=mean(storc(i:N:end,9)-storc(i:N:end,8));
    sat(i,6)=std(storc(i:N:end,9)-storc(i:N:end,8));
    sat(i,7)=prctile((storc(i:N:end,9)-storc(i:N:end,8)),75)-prctile((storc(i:N:end,9)-storc(i:N:end,8)),25);
    h=4*i-1;
    Fanal(:,[i i+N])=[vec(storb(:,h:4*N:end)) vec(storb(:,h+1:4*N:end))];
end

%%
Cf=[];
for h=1:N
    [a p]=corrcoef(Fanal(:,[h h+N]));
    Cf(h,1:2)=[a(1,2) p(1,2)];
end

%%
save ivol4
% ivol3 is with [.005 .05 .20 .33 .4 .5 .75]

