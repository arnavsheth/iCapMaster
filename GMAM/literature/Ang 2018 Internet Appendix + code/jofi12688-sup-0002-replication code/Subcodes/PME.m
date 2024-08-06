function P=PME(Factors,Out,In)
T=size(Factors,1);
disc=ones(T+1,1);
for ci=2:T+1
    disc(ci,1)=disc(ci-1,1)/(1+Factors(ci-1,:));
end
P=zeros(size(Out,2),1);
for ci=1:size(Out,2)
 P(ci,1)=sum(Out(:,ci).*disc)/sum(In(:,ci).*disc);
end

end