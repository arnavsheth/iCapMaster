% Search for alpha that gets the EW av PMEs to be unity

function mPME=MPME_simple(F,Out,In)
T=size(RF,1);
disc=ones(T+1,1);
for c=2:T+1
    disc(c,1)=disc(c-1,1)/(1+F(c-1));
end
pme=zeros(size(Out,2),1);
for i=1:size(Out,2)
 pme(i,1)=sum(Out(:,i).*disc)/sum(In(:,i).*disc);
end
mPME=(sum(pme.*weight)/sum(weight)-1)^2;
end

