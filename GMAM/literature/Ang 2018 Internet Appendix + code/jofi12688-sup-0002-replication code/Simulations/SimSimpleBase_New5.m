clear

% This is for referee 2 - Case where it should work nicely
alphaTrue=0.01;phiTrue=0;isfactor=1;
alpha=alphaTrue;phi=phiTrue;
nfund_b=5;nfund_n=3;
wc=1;
Burn = 500;Num_Sim = 10000;
sellfreq=1/28;fi=1;

for roud=1:50
    rng('shuffle')
    Factors=.02+.1*randn(80,1);
    epsilon=randn(80,1);
    
    %for sigmaf=[.025 .1 .2]
        for sigmaf=[.1]
    vv=sigmaf*randn(80,1);
    f=filter(1,[1 -phiTrue],vv);
    
    %for beta=[.5 1.5]
        for beta=[1.5]
    g=alphaTrue+beta*Factors+f;
    truth=[mean(g) std(g) sacf(g,2)'];
    Ret=repmat(g',100000,1);
    p_m_alpha=0.01;p_m_bta=beta;
    
    %for Se=[.1 .5]
    for Se=[.25]        
        nfund_base=nfund_b*fi;
        nfund_next=nfund_n*fi;
        Inv=repmat(eye(20),nfund_base,1);Div=[];Value=[];trueret=[];
        n=size(Inv,1);a=find(Inv(:,1)>0);Value(a,1)=Inv(a,1);time=80;
        
        %% Generate cash flows for each investment
        for q=2:time
            a=find(Value(:,q-1)>0);
            na=size(a,1);ns=round(na*sellfreq);
            R=1+Ret(a,q-1);
            Value(a,q)=Value(a,q-1).*R;
            sell=a(1:ns); % First in, first out
            % Compute true return
            trueret(q-1)=sum(Value(a,q))/sum(Value(a,q-1))-1;
            % Stocks that are sold pay a final dividend
            Div(sell,q)=Value(sell,q);Value(sell,q:time+1)=-99;
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
        if fi>1
            for i=1:20:size(Div,1)-1
                Invest(:,nf)=sum(Inv(i:i+19,:))';
                Divest(:,nf)=sum(Div(i:i+19,:))';
                nf=nf+1;
            end
        else
            for i=1:20:size(Div,1)-1
                for h=0:3
                    Invest(:,nf)=sum(Inv(i+h:4:i+19,:))';
                    Divest(:,nf)=sum(Div(i+h:4:i+19,:))';
                    nf=nf+1;
                end
            end
        end

        T=size(Div,2)-1;
        icol=1;wfactor=1;
        Num_Factors=1;RF=zeros(T,1);
        Num_Funds=size(Divest,2);
        
        u=Se*randn(Num_Funds,1);
        error=repmat(exp(u)',T+1,1);
        div=Divest./error;
        Inv=Invest;
        CF_mat=div-Inv;
        DivAll=div;InvAll=Inv;fsize=ones(cols(div),1);

                
        %% Sub-code here: Specify priors and the like
        Parameter %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %% Simulation Loop
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('starting with burning')
        disp([roud pi])
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
            thistheta = [alpha phi bta' sig_g sig r2];               
            theta_mat = [theta_mat; thistheta];
            if abs(j/2000-round(j/2000))<0.0001
                bb=[trueret' g E_g_ret];cc=corrcoef(bb)
                mean(theta_mat(:,3))
            end
        end
        
        output
        Storoutput(wc,16:18) =[mean(G) mean(F) mean(E_g_ret)];
        E_g_ret=G;
        
        bb=[trueret' E_g_ret];
        cc=corrcoef(bb);m=mean(bb);sb=std(bb);sk=skewness(bb);k=kurtosis(bb);
        error=bb(:,1)-bb(:,2);
       
        %x=[ones(80,1) bb(:,2)];ab=ols(bb(:,1),x);
        x=[ones(80,1) Factors];ab=ols(bb(:,1),x);
        
        all=[m sb sk k];
        beep
        wc
        Storoutput(wc,11:14)=mean(theta_mat(:,1:4)); % alpha phi bta' sigma_f
        Storoutput(wc,5:8)=[sb cc(2,1) 100*mean(error.^2)]; % est. std(g), corr, MSE

        %tstatStruct=regstats(trueret',E_g_ret,'linear','tstat');
        %Storoutput(wc,7:8)=tstatStruct.tstat.beta';
        %Storoutput(wc,9) = (tstatStruct.tstat.beta(2)-1)./tstatStruct.tstat.se(2);
        
        Storoutput(wc,1:4)=[beta sigmaf Se ab.beta(2)];
        
        Storoutput(wc,21:24)=all(1:2:end); % 1.true mean(g), 2. true std(g), 3. true skewness(g) 4. true kurtosis(g)
        Storoutput(wc,25:28)=truth;
        
        Storoutput(wc, 29) = mean(mpme_arr);  % estimation of lump-sum mean(log(PME)) 
        Storoutput(wc, 30) = std(mpme_arr);
        Storoutput(wc, 31) = std(mean(log(PME_arr)'));

    wc=wc+1;
    end
    end
    end   
    end

% N=15;
% for i=1:N
%     sa(i,:)=mean(satot(i:N:end,:));
% end

% N=32;
% for i=1:N
%     sa(i,:)=mean(Storoutput(i:N:end,:));
% end

%save SimSimpleBase_New5

