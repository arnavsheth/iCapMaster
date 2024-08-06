using Revise
using Distributions, Statistics, LinearAlgebra, UnPack
import emf_gmam_ae as m

#simulats the data generating process
function simulateseries(; dims, 
    Ktrue=dims.K*PARAM[:simulateKtruefrac] |> floor |> Int, 
    β=[PARAM[:simulatebetatrueintercept]; 
      fill(PARAM[:simulatebetatrueselected], Ktrue); 
      fill(PARAM[:simulatebetatruenotselected],length((Ktrue+2):(dims.K)));],
    W=rand(Normal(0,1),dims.K-1, dims.K-1) |> Wraw->Wraw./vec(sum(Wraw, dims=2)),
    ϕ=[p for p ∈ 1:dims.P]/sum(1:(dims.P+1)),
    σ2x=PARAM[:simulatesigma2residx],
    σ2y=PARAM[:simulatesigma2residy],
    σ2xhat=PARAM[:simulatesigma2xhat],
    σ2r=PARAM[:simulatesigma2r],
    Er=PARAM[:simulateEr],
    br=PARAM[:simulaterpersistance],
    ν=PARAM[:simulatenu],
    F = [ones(dims.T) rand(Normal(),dims.T,dims.K-1)*W],
    testsimulateseries = PARAM[:testsimulateseries]
  )

  #need to rescale F
  F[:,2:end] .*= σ2xhat^0.5/(β[2:end]'cov(F[:,2:end])*β[2:end])^0.5

  @unpack T,K,P,S = dims

  @assert 1.0 > sum(ϕ)
  @assert Ktrue < K

  #interest rates are highly persistant
  σ2re = (1-br^2)*σ2r
  r = rand(Normal(0.0,σ2re^0.5),T)
  for t ∈ 2:T
    r[t] += r[t-1]*br
  end
  #do not cross the ZLB
  r .= (r .+ Er) .|> rt->max(rt,0.0) 

  #x = F*β .+ rand(Normal(0,σ2x^0.5),T)
  x = F*β .+ rand(TDist(ν),T) .* ((ν-2)/ν*σ2x)^0.5 .+ r

  #@eval Main x=$x

  y = m.formX̃L(;dims, x)*ϕ .+ x[dims.s2t] .+ rand(Normal(0, σ2y^0.5),S) 

  s = (;y, x, F, r, β, ϕ)

  if testsimulateseries
    @eval Main s=$s
  end

  #throw("stop")

  return s 
end

#this loads the set of test data
function loadmcmctest(;
    priorset=PARAM[:simulatepriorset],
    mcmcinitmethod=PARAM[:defmcmcinitmethod],
    model=MODEL_INDEX[PARAM[:simulatemodel]]
    )
  
  K=PARAM[:simulateK]
  S=PARAM[:simulateS] 
  P=PARAM[:simulateP]
  Δt=PARAM[:simulatefrequency]
  Fβs = ["intercept"; ["β[$k]" for k in 2:K]] .|> Symbol
  dims=m.Dims(; K,S,P,Δt,Fβs)

  @unpack F,y, r = simulateseries(; dims)
  data=Data(;F, y, r)


  #@info "Simulating model $model given priorset $priorset"
  hyper=loadpriorsfromasciidict(PARAM[priorset], model; dims, data)

  
  Θ=initmcmc(mcmcinitmethod, model;data,hyper,dims)

  testΦ(rand(P); dims)
  
  records = ChainRecords(PARAM[:simulatemcmcfieldstocapture]; Θ, chainid=:simmcmc, 
    testchainrecords=true,
    burnt=false,
    numsamplerecords=PARAM[:simulatemcmcnumsamplerecords],
    numburnrecords=PARAM[:simulatemcmcnumburnrecords],
    numchains=PARAM[:simulatemcmcnumchains])

  dgp= DGP(;Θ,data, dims, hyper, records)

  return dgp


end



#testsimulateseries()