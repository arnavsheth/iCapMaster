clear
Nsimu = 10000000; % Number of MC simu
Ninv = 10; % Number of investments
vol=.5/2;
% PV_div_over_PV_inv = NaN(1,Nsimu); % will contain PME in every simu
% mat = [5*ones(Nfunds/5,1);10*ones(Nfunds/5,1);8*ones(Nfunds/5,1);3*ones(Nfunds/5,1);7*ones(Nfunds/5,1)];
% maturities of funds

%weight=[.2 .8/(Ninv-1)*ones(1,Ninv-1)];
duration=[40 16*ones(1,Ninv-1)];
weight=[1/Ninv*ones(1,Ninv)];
%duration=[16*ones(1,Ninv)];

for k=1:Nsimu
    eps =  randn(Ninv,max(duration))*vol-.5*vol^2;
    U(1)=exp(sum(eps(1,1:duration(1))));
    U(2:Ninv)=exp(sum(eps(2:Ninv,1:duration(2)),2));
    out(k)=sum(weight.*U);
end

log_PME = log(out); % take the log and compare to normal RV
histfit(log_PME)
std(log_PME)
mean(log_PME)
[H,P,stat]=jbtest(log_PME)

%weight=[.2 .8/(Ninv-1)*ones(1,Ninv-1)];
%duration=[40 16*ones(1,Ninv-1)];
weight=[1/Ninv*ones(1,Ninv)];
duration=[16*ones(1,Ninv)];

for k=1:Nsimu
    eps =  randn(Ninv,max(duration))*vol-.5*vol^2;
    U(1)=exp(sum(eps(1,1:duration(1))));
    U(2:Ninv)=exp(sum(eps(2:Ninv,1:duration(2)),2));
    out(k)=sum(weight.*U);
end

log_PME = log(out); % take the log and compare to normal RV
histfit(log_PME)
std(log_PME)
mean(log_PME)
[H,P,stat]=jbtest(log_PME)

