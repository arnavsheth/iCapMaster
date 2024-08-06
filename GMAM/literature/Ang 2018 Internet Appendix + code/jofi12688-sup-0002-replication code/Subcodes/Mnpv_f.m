function mnpv=Mnpv_f(alpha,bta,Factors,RF,f,Out,In)
T=size(RF,1);
disc=ones(T+1,1);
for c=2:T+1
    disc(c,1)=disc(c-1,1)/(1+RF(c-1,1)+sum(Factors(c-1,:).*bta)+alpha+f(c-1));
end
pv=zeros(size(Out,2),1);
for i=1:size(Out,2)
 pv(i,1)=sum(Out(:,i).*disc)-sum(In(:,i).*disc);
end
mnpv=mean(pv)^2;
end

