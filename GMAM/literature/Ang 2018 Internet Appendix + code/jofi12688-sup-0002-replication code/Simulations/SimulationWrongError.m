clear
% This is what run on OSC, quick version
wi=1;
Burn = 1000;Num_Sim = 3000;
ivola=[0.25 0.05];
vol=[0.05 0.05];
roa=[0.5 0];
feesa=[0 1];
facta=[1 0];
nfunda=[1 2];
fewinva=[0 1];
sellw=[0 1];
sellf=[1/28 1/4];
dyielda=[0 .01];
rprior=[1 0];

parameters=[ivola;roa;feesa;facta;nfunda;fewinva;sellw;sellf;dyielda;rprior;vol];
np=size(parameters,1);

for nround=1:30 % number of times the simulation is run
    par=[];rng('shuffle')
    % default return series
    mkt=.02+.1*randn(80,1);
    vvrnd=randn(80,1);
    dRetRnd=randn(20000,80);

    %for si=0:np %%% To be changed with 'np'
    for si=0:2 %%% To be removed
        h=ones(np,1);if si>0;h(si)=2;end
        for j=1:np
            par(j)=parameters(j,h(j));
        end
        c=1;
        ivol=par(c);c=c+1;
        ro=par(c);c=c+1;
        fees=par(c);c=c+1;
        isfactor=par(c);c=c+1;
        nfund=par(c);c=c+1;
        fewinv=par(c);c=c+1;
        sellwinners=par(c);c=c+1;
        sellfreq=par(c);c=c+1;
        dyield=par(c);c=c+1;
        prior=par(c);c=c+1;
        volall=par(c);c=c+1;
        
        T=80;f=filter(1,[1,-ro],volall*vvrnd);
        
        if prior==1;p_m_alpha=0.01;p_m_bta=1.5;else;p_m_alpha=0;p_m_bta=1;end        
        if fewinv==1;nfund_base=10/nfund;nfund_next=2/nfund;else;nfund_base=40/nfund;nfund_next=8/nfund;end
        if fees==1;mngtfees=0.01;carry=0.2;else;mngtfees=0;carry=0;end
        if isfactor==1;Factors=mkt;else;Factors=zeros(T,1);p_m_bta=0;end

        tret=.01+f+Factors*1.5;
        Ret=repmat(tret',20000,1)+ivol*dRetRnd;
        
        Inv=repmat(eye(20),nfund_base,1);Div=[];Value=[];trueret=[];
        n=size(Inv,1);a=find(Inv(:,1)>0);Value(a,1)=Inv(a,1);time=80;
        
        %% Generate cash flows
        for q=2:time
            a=find(Value(:,q-1)>0);
            na=size(a,1);ns=round(na*sellfreq);
            R=1+max([Ret(a,q-1) -0.99*ones(na,1)],[],2);
            
            Value(a,q)=Value(a,q-1).*R*(1-mngtfees);
            
            % Sell 'freqsell' stocks that are alive every quarter - controls average life
            if sellwinners==1     % Sell winners
                PastRet=mean(Ret(a,max([q-16 1]):q-1),2); % Select investments sold based on past two years returns
                b=find(PastRet>prctile(PastRet,100-150*sellfreq));sell=a(b(1:ns));
            else
                sell=a(1:ns); % First in, first out
            end
            
            % Carried interest when selling
            w=find(Value(sell,q)>1.08^4);Value(sell(w),q)=1+(Value(sell(w),q)-1)*(1-carry);
            
            % Compute true return
            trueret(q-1)=sum(Value(a,q))/sum(Value(a,q-1))-1;
            
            %% Stocks that are sold pay a final dividend
            Div(sell,q)=Value(sell,q);Value(sell,q:time+1)=-99;
            % Those not sold pay an intermediary dividend
            a=find(Value(:,q)>0);Div(a,q)=dyield*Value(a,q);Value(a,q)=(1-dyield)*Value(a,q);
            
            % Last 20 quarters
            if q<=time-19
                Inv=[Inv zeros(size(Inv,1),1);zeros(20*nfund_next,q-1) repmat(eye(20),nfund_next,1)];
            end
            a=find(Inv(:,q)>0);Value(a,q)=Inv(a,q); % Value at the beginning of quarter q is the sum of all investments
            
        end
        
        %% Quarterly cash flows are done. Now time T: All stocks are sold at the end
        q=time+1;
        a=find(Value(:,q-1)>0);
        R=1+max([Ret(a,q-1) -0.99*ones(size(a,1),1)],[],2);
        Value(a,q)=Value(a,q-1).*R;
        trueret(q-1)=sum(Value(a,q))/sum(Value(a,q-1))-1;
        Div(a,q)=Value(a,q);Inv(:,q)=0;
        
        %% Pool investments to form funds
        nf=1;Invest=[];Divest=[];
        for i=1:20:size(Div,1)-1
            if fewinv==1
                for h=0:3
                    Invest(:,nf)=sum(Inv(i+h:4:i+19,:))';
                    Divest(:,nf)=sum(Div(i+h:4:i+19,:))';
                    nf=nf+1;
                end
            else
                Invest(:,nf)=sum(Inv(i:i+19,:))';
                Divest(:,nf)=sum(Div(i:i+19,:))';
                nf=nf+1;
            end
        end
        
        %% Regression to compute true alpha, beta and phi
        T=size(Div,2)-1;wc=1;icol=1;wfactor=1;Num_Factors=1;RF=zeros(T,1);
        y=trueret';aa=autocorr(y);x=[ones(80,1) mkt];as=ols(y,x);
        
        %% Sub-code here: Specify priors and the like
        div=Divest;Inv=Invest;
        DivAll=Divest;InvAll=Invest;Num_Funds=size(Divest,2);
        CF_mat=Divest-Invest;
        %AnyparamQtryWrongError
        Parameter %%%%%%%%%%%%%
        

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
            if abs(j/2000-round(j/2000))<0.0001
                bb=[trueret' E_g_ret];cc=corrcoef(bb)
            end
        end
        bb=[trueret' E_g_ret];
        %x=[ones(80,1) bb(:,2)];ab=ols(bb(:,1),x);
        x=[ones(80,1) mkt];ab=ols(bb(:,1),x);
        dr=repmat(cumprod([1;1./(1+trueret')])',size(Divest,2),1)';
        PVD=sum(Divest.*dr);PVI=sum(Invest.*dr);PMEt=(PVD./PVI)';
        
        thistheta = [alpha bta' phi sig_g alpha B sig r2];
        cc=corrcoef(bb);m=mean(bb);sb=std(bb);
        disp('volatility: true vs estimated:');sb
        disp('Correlation: true/actual & estimated:');[wi cc(2,1)]
        error=bb(:,1)-bb(:,2);
        all=[m sb];
        %beep
        Storoutput(wi,1:2)=all(1:2:end); % 1.true mean(g), 2. true std(g), 3. true skewness(g) 4. true kurtosis(g)
        Storoutput(wi,3:4)=[ab.beta(2) std(log(PMEt))];
        Storoutput(wi,6:12)=[all(2:2:end) 0 cc(2,1) mean(error.^2) mean(abs(error)) sacf(f,1)]; % 6. est. mean(g) 7. est. std(g) 8. est skew(g) 9. est kurt(g) 11. corr(true_g, est_g) 12. MSE 13.MAD 14. true Autocorr
        Storoutput(wi,26:30)=mean(theta_mat(:,1:5)); % 26. alpha_full_real 27. bta' 29. phi 30. sig_g
        Storoutput(wi,33:37)=std(theta_mat(:,1:5)); % Std of these
        Storoutput(wi,15:25)=[ivol volall ro fees isfactor nfund fewinv sellwinners sellfreq dyield prior];

        Storoutput(wi,48)=si; % which combination this was
        Storoutput(wi,49)=mpme;
        wi=wi+1;
    end
end
Tabl=[];
Storey=[];cc=np+1;
for h=1:cc
    Storey(h,:)=prctile(Storoutput(h:cc:end,:),25);
    Storey(h+cc,:)=prctile(Storoutput(h:cc:end,:),50);
    Storey(h+2*cc,:)=prctile(Storoutput(h:cc:end,:),75);
    Storey(h+3*cc,:)=mean(Storoutput(h:cc:end,:));
end
s=1;
Tabl(1:cc,s:3:12)=Storey((s-1)*cc+1:s*cc,[26:28 49]);s=s+1;
Tabl(1:cc,s:3:12)=Storey((s-1)*cc+1:s*cc,[26:28 49]);s=s+1;
Tabl(1:cc,s:3:12)=Storey((s-1)*cc+1:s*cc,[26:28 49]);s=s+1;
Tabl(1:cc,1:3)=Tabl(1:cc,1:3)*4;

Tabl(np+2:2*np+2,[1 3])=Storey((s-1)*cc+1:s*cc,[1 6])*4;
Tabl(np+2:2*np+2,[2 4])=Storey((s-1)*cc+1:s*cc,[2 7])*2;
Tabl(np+2:2*np+2,5)=Storey((s-1)*cc+1:s*cc,9);
Tabl(np+2:2*np+2,6)=Storey((s-1)*cc+1:s*cc,10)*100;

N=3;
for i=1:N
    sa(i,:)=mean(Storoutput(i:N:end,:));
end

%save SimWrongE_2a