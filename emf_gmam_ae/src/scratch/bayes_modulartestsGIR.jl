#this file is designed to check the algebra of the Bayesian calculations
#While snippits may be used in the poc, this is not designed to be used without major refactoring in the poc
#the primary purpose is to test the algebra

using Revise
import emf_gmam_ae as m
using Statistics, StatsBase, LinearAlgebra, Distributions, SparseArrays, SpecialFunctions
using Random, DataFrames, StatsBase, BenchmarkTools, UnPack, ProgressMeter

#throw("use the truncated gamma function as a the prior")
Random.seed!(1111)


lpdist(args...;kwargs...)=m.lpdist(args...;kwargs...)

function gencorrelateddata(;rows,cols,nfactors=cols)

  #this will weight the cols into linear combinations
  W = rand(cols, nfactors)

  uncorrelated = rand(TDist(5.), rows, cols)
  err = rand(TDist(5.), rows, cols)*cols/4
  correlated = uncorrelated*W .+ err

  return correlated
end




function formtestdata(; 
  S=5, 
  P=3,Δt=3, 
  K=4,
  nsim=10^4,
  fieldstocapture=[:x,:ϕ,:β,:γ,:ω,:τy,:τx,:ψ,:ν, :τβ, :τϕ ],
  numsamplerecords=1000,
  numburnrecords=200,
  numchains=1,)


  dims=m.Dims(;S,P,K,Δt,testdims=true)

  @unpack T = dims
  @assert T == S*Δt+P


  #sim some data
  Fr = [ones(T) gencorrelateddata(;rows=T,cols=K)]
  F = Fr[:,1:(end-1)]
  r = Fr[:,end]

  #hyperparameters- some of these are set in advance due to dependencies
  ζν0 = (4+sqrt(816))/400 #corresponds to mode 4, variance 200
  αν0 = 4*ζν0+1  
  #set the max integration limit at machine precision
  νmax = Inf #100#quantile(Gamma(αν0,1/ζν0),1.0-eps())
  νmin = 0.00
  rνdist=Gamma(αν0, 1/ζν0) #set proposal to prior
  lrν=(x)-> logpdf(rνdist,x)

  hyper = m.DGPModelParametersGIR(;
    x = (;),  
    ϕ = (;
      ϕ0 = rand(Uniform(-1.0,1.0), P),
      M0 = gencorrelateddata(;rows=nsim, cols=P) |> cov |> inv,
      ϕmin=nothing,
      ϕmax=nothing, ), 
    β = (;
      v=10.0^-1.0, #switching this between ^-1 and ^-2 creates good test cases for AO::Matrix and A0::Diagonal respectivelly
      β0 = 1:K .|> k->Float64(k)*(-1)^k,#ones(K),
      βΔ0 = K:-1:1 .|> k->Float64(k)*(-1)^(k+1),#ones(K),     
      #A0 = [0.1 for k ∈ 1:K] |> Diagonal |> Matrix{Float64}),
      A0 = gencorrelateddata(;rows=nsim, cols=K) |> X->X'X ./ (T)),      

    γ = (;),
    ω = (;
      κ0=1.0,
      δ0=1.0), 
    τϕ=(;
      αϕ0=1.5,
      ζϕ0=dims.S/2),      
    τβ=(;
      αβ0=1.5,
      ζβ0=dims.T/2),
      τy = (;
      αy0=1.1,
      ζy0=100.),      
    τx = (;
      αx0=1.1,
      ζx0=5.0),   
    ψ = (;),
    ν = (;
      νmax,
      νmin,
      ζν0,
      αν0,   
      rνdist, #set proposal to prior
      lrν,
  ))


  #test parameter values
  ϕtest = hyper.ϕ0 .+ rand(Uniform(-1.0,1.0), P)
  ψtest = 1 ./ (rand(Normal(1.0,1.0), T) .^ 2 .+ 0.1)
  νtest = 4.0
  ωtest = 0.3
  τxtest = 0.2
  τytest = 0.3
  γtest = [true; true; false; rand([false,true], K-3)]
  βtest = (γtest + (1.0.-γtest).*0.01) .* abs.(hyper.βΔ0) .+ hyper.β0
  τβtest = 10.0
  τϕtest = 5.0


  xtest = F*βtest + r + rand(Normal(0,sqrt(K)), T)

  Θ = m.DGPModelParametersGIR(;x=xtest, ϕ=ϕtest, β=βtest, γ=γtest, ω=ωtest, τy=τytest, τx=τxtest, 
    ψ=ψtest, ν=νtest, τβ=τβtest, τϕ=τϕtest)
  
  m.testΦ(ϕtest; dims)

  data=m.Data(;
    F,
    r,
    y=m.formΦ(Θ.ϕ; dims)*Θ.x .+ rand(Normal(0.0,sqrt(P)), S))
  

    records = m.ChainRecords(fieldstocapture,; Θ, chainid=:_,  burnt=false,
    numsamplerecords, numburnrecords, numchains,)

  dgp= m.DGP(;Θ,data, dims, hyper, records)

  return dgp
end






function testpostϕ(dgp::m.DGP{<:m.DGPModelParametersG}; iter)
  @info "testing ϕ (iter=$iter)"
  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack S,P,T,R = dims
  @unpack ϕ0, M0, αϕ0, ζϕ0, αβ0, ζβ0, ϕmin, ϕmax  = hyper
  @unpack x, ϕ, τy, τx, ψ,τϕ,τβ = Θ
  @unpack y = data

  #we will get the posterior density and check it against the derived conditional posterior density
  post = lpdist(dgp,:post)
  X̃L = m.formX̃L(;x, dims)
  X̃L_alt = m._formXL_alt(x;dims)*R
  ỹ = m.formỹ(y, x;dims)
  @assert all((X̃L .≈ X̃L_alt) .| ((X̃L .+ 1.0) .≈ (X̃L_alt .+ 1.0)))

  #verify the prior and likelihood
  @assert lpdist(dgp,:ϕprior) ≈ (-(τy/2)*(ϕ-ϕ0)'*M0*(ϕ-ϕ0).*τϕ +0.5*(P*log(τy)-P*log(2π)+log(det(M0))+P*log(τϕ)))
  @assert lpdist(dgp,:ylike) ≈ (-(τy/2)*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ) +0.5*(S*log(τy)-S*log(2π)))
  @assert lpdist(dgp,:τϕprior) ≈ (αϕ0*log(ζϕ0)-loggamma(αϕ0) + (αϕ0-1)*log(τϕ)-ζϕ0*τϕ)
  @assert lpdist(dgp,:τβprior) ≈ (αβ0*log(ζβ0)-loggamma(αβ0) + (αβ0-1)*log(τβ)-ζβ0*τβ )    

  #verify the algebra from the modelnotes documentation
  #step 1
  c1 = ((S+P)/2*log(τy/(2π))+0.5*log(det(M0))+P/2*log(τϕ)
    +lpdist(dgp, :xprior)+lpdist(dgp, :βprior)+lpdist(dgp, :γprior)+lpdist(dgp, :ωprior)
    +lpdist(dgp, :τyprior)+lpdist(dgp, :τxprior)+lpdist(dgp, :ψprior)+lpdist(dgp, :νprior)
    +lpdist(dgp, :τϕprior)+lpdist(dgp, :τβprior)
    )
  lqϕ1Mc1 = -τy/2*((ỹ-X̃L*ϕ)'* (ỹ-X̃L*ϕ)+(ϕ-ϕ0)'*M0*(ϕ-ϕ0).*τϕ)
  @assert c1 + lqϕ1Mc1 ≈ post 
 
  lqϕ2Mc2 = -τy/2*(ϕ'*(X̃L'*X̃L+(M0.*τϕ))*ϕ-(ỹ'*X̃L+ϕ0'*(M0.*τϕ))*ϕ - ϕ'*(X̃L'*ỹ+(M0*ϕ0.*τϕ)))
  c2 = c1-τy/2*(ỹ'*ỹ+ϕ0'*M0*ϕ0.*τϕ)
  @assert c2 + lqϕ2Mc2 ≈ post  

  #now we can test the updating parameters
  Λϕ = τy*(X̃L'*X̃L+M0.*τϕ)
  μϕ = τy*inv(Λϕ)*(X̃L'*ỹ+M0*ϕ0.*τϕ)

  @assert Λϕ ≈ m.updateθpϕ(dgp).Λϕ
  @assert μϕ ≈ m.updateθpϕ(dgp).μϕ

  #test the conditional draw
  Random.seed!(11)
  manualdraw = m.draw(:ϕ;μϕ, Λϕ, Σϕ=m.pdinv(Λϕ), ϕmin, ϕmax), 
  Random.seed!(11)
  conddraw = m.conditionaldraw(:ϕ; dgp).Θ.ϕ
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

  lqϕ3Mc3 = -0.5*(ϕ-μϕ)'*Λϕ*(ϕ-μϕ)
  c3 = c2+0.5*(μϕ'*Λϕ*μϕ)
  @assert c3 + lqϕ3Mc3 ≈ post

  lqϕ4Mc4 = logpdf(MultivariateNormal(μϕ,inv(Λϕ) |> Symmetric),ϕ)
  c4 = c3+P/2*log(2π)-0.5*log(det(Λϕ))
  @assert c4 + lqϕ4Mc4 ≈ post


  ##Now need to check the expectations
  #need placeholders for the updates
  μx=x.-0.1
  Λx=τy*τx .* m.formΨ(ψ)
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  αϕ=hyper.αϕ0-0.2
  ζϕ=hyper.ζϕ0-0.1

  vbx = m.VBqx(dgp; μx, Λx, testmoments=true)
  @unpack EXL, EXLtXL, EX̃LtX̃L, EX̃L, Eỹ, EX̃Ltỹ = vbx

  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy

  vbτϕ = m.VBqτϕ(dgp; αϕ, ζϕ, testmoments=true)
  @unpack Eτϕ = vbτϕ  

  dgpmom = m.DGP(dgp; x=vbx, τy=vbτy, τϕ=vbτϕ , strict=false)

  @unpack μϕ,Λϕ = m.updateθqϕ(dgpmom, testmoments=true)
  @assert Λϕ ≈ Eτy*(EX̃LtX̃L+M0 .* Eτϕ)
  @assert μϕ ≈ Eτy*inv(Λϕ)*(EX̃Ltỹ+M0*ϕ0 .* Eτϕ)
  lqϕθ = -0.5*(ϕ-μϕ)'*Λϕ*(ϕ-μϕ)+0.5*(μϕ'*Λϕ*μϕ)

  lqϕθsim = 0.0
  τϕsim = 0.0


  for i ∈ 1:iter
    dgpi = m.DGP(dgp; x=m.draw(:x;μx, Λx), τy=m.draw(:τy, ;αy,ζy), τϕ=m.draw(:τϕ; αϕ, ζϕ,))
    sim = m.updateθpϕ(dgpi)
    lqϕθsimi = -0.5*(ϕ-sim.μϕ)'*sim.Λϕ*(ϕ-sim.μϕ) + 0.5*sim.μϕ'*sim.Λϕ*sim.μϕ
    lqϕθsim += lqϕθsimi
    τϕsim += dgpi.Θ.τϕ
  end

  lqϕθsim /= iter
  τϕsim /= iter

  #compare the results
  @info "*****lqϕθ: $lqϕθ; lqθsim: $lqϕθsim"
  @info "*****Eτϕ: $Eτϕ; τϕsim: $τϕsim"



end

function testpostx(dgp; iter)
  @info "testing x (iter=$iter)"

  @unpack   data, Θ, dims,hyper = dgp
  @unpack S,P,T,K,Δt = dims
  @unpack M0, ϕ0, ϕmin, ϕmax = hyper
  @unpack x, ϕ, β, τx, τy, ψ, ν = Θ
  @unpack y, F, r = data
  
  X̃L = m.formX̃L(;x, dims)
  Φ = m.formΦ(ϕ; dims)
  @assert all(X̃L*ϕ .+ sum(x[dims.s2t.-l] for l ∈ 0:(Δt-1)) .≈ Φ*x)


  Ψ = ψ |> m.formΨ
  post = lpdist(dgp, :post)

  #verify the prior and likelihood
  @assert lpdist(dgp,:xprior) ≈ (-(τy*τx/2/T)*((x-r)*sqrt(T)-F*β)'*Ψ*((x-r)*sqrt(T)-F*β) +0.5*(T*log(τy*τx)-T*log(2π*T)+log(det(Ψ))))
  @assert lpdist(dgp,:ylike) ≈ (-(τy/2)*(y-Φ*x)'*(y-Φ*x) +0.5*(S*log(τy)-S*log(2π)))

  #walk through the algebra for the updating equations, starting with
  #the conditional equations (we'll work on the VB moment equations later)
  lqx1Mc1 = -τy/2*((y-Φ*x)'*(y-Φ*x) + τx/T*((x-r)*sqrt(T)-F*β)'*Ψ*((x-r)*sqrt(T)-F*β))
  c1 = ((S+T)/2*log(τy/(2π))+T/2*log(τx)+0.5*log(det(Ψ))-T/2*log(T)
    +lpdist(dgp, :ϕprior)+lpdist(dgp, :βprior)+lpdist(dgp, :γprior)+lpdist(dgp, :ωprior)
    +lpdist(dgp, :τyprior)+lpdist(dgp, :τxprior)+lpdist(dgp, :ψprior)+lpdist(dgp, :νprior)
    +lpdist(dgp, :τϕprior)+lpdist(dgp, :τβprior)
    )
  
  @eval Main (lqx1Mc1,c1,post,dgp,Ψ) = ($lqx1Mc1,$c1,$post,$dgp,$Ψ)
  @assert lqx1Mc1 + c1 ≈ post

  lqx2Mc2=-τy/2*(x'*(Φ'*Φ + τx .* Ψ)*x - (y'*Φ + τx/sqrt(T) * (β'*F'+sqrt(T)*r')*Ψ)*x-x'*(Φ'*y + τx/sqrt(T).*Ψ*(F*β+sqrt(T)*r)))
  c2 = c1  - τy/2*y'*y - τx*τy/2/T*(β'*F'+sqrt(T)*r')*Ψ*(F*β+sqrt(T)*r)
  @assert lqx2Mc2 + c2 ≈ post

  Λx=τy.*(Φ'*Φ+τx.*Ψ)
  μx=τy.*inv(Λx)*(Φ'*y+τx/sqrt(T)*Ψ*(F*β+r*sqrt(T)))

  @assert μx ≈ m.updateθpx(dgp).μx
  @assert Λx ≈ m.updateθpx(dgp).Λx

  #test the conditional draw
  Random.seed!(11)
  manualdraw = m.draw(:x; μx, Σx=m.pdinv(Λx))
  Random.seed!(11)
  conddraw = m.conditionaldraw(:x; dgp).Θ.x
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"
  
  lqx3Mc3 = -0.5*(x-μx)'*Λx*(x-μx)
  c3 = c2+μx'*Λx*μx/2
  @assert lqx3Mc3 + c3 ≈ post

  lqx4Mc4 = logpdf(MultivariateNormal(μx,Λx |> m.pdinv),x)
  c4 = c3+T/2*log(2π)-0.5*log(det(Λx))
  @assert lqx4Mc4 + c4 ≈ post  

  #now investigate the moments via montecarlo
  μϕ=ϕ
  Λϕ=τy .* M0
  #Σϕ=Λϕ|>m.pdinv
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  αx=hyper.αx0+0.4
  ζx=hyper.ζx0+0.3
  #ν = (hyper.νmin + hyper.νmax)/2
  αψ=[ν/2 for i ∈ 1:T]
  ζψ=[ν/2 for i ∈ 1:T]
  D = m.formD(Θ.γ,hyper.v)
  μβ= D*β
  Λβ = (τx*τy .* D*hyper.A0*D)*10

  
  EΨ = m.moment(:EΨ; αψ, ζψ)

  vbϕ = m.VBqϕ(dgp; μϕ, Λϕ, testmoments=true)
  @unpack EΦ,EΦtΦ = vbϕ
  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack μβ= vbβ
  vbψ = m.VBqψ(dgp;  αψ, ζψ, testmoments=true)
  @unpack EΨ= vbψ  

  dgpmom = m.DGP(dgp; ϕ=vbϕ, τy=vbτy, τx=vbτx, 
    ψ=vbψ, β=vbβ, strict=false)
  @unpack μx,Λx = m.updateθqx(dgpmom)
  @assert Λx ≈ Eτy*(EΦtΦ+Eτx*EΨ) "sum(Λx)=$(Λx) while sum(Eτy*(EΦtΦ+Eτx*EΨ)) =\n $(sum(Eτy*(EΦtΦ+Eτx*EΨ)))!
   \n EΦtΦ=$EΦtΦ, \n EΨ=$EΨ"
  @assert μx ≈ Eτy*inv(Λx)*(EΦ'y+Eτx/sqrt(T)*EΨ*(F*μβ+r*sqrt(T)))
  lqxθ = -0.5*(x-μx)'*Λx*(x-μx) + μx'*Λx*μx/2
  #lqxθ=-Eτy/2*(x'*(EΦtΦ + Eτx .* EΨ)*x - (y'*EΦ + Eτx * μβ'*F'*EΨ)*x-x'*(EΦ'*y + Eτx.*EΨ*F*μβ))

  sim = Dict(:ΦtΦ=>zeros(T,T), :Φ=>zeros(S,T), :τx=>0.0, :τy=>0.0, :ψ=>zeros(T), :β=>zeros(K))
  lqxθsim = 0.0

  for i ∈ 1:iter
    dgpi = m.DGP(dgp; ϕ=m.draw(:ϕ;μϕ, Λϕ, Σϕ=m.pdinv(Λϕ), ϕmin, ϕmax),  
    τy=m.draw(:τy, ;αy,ζy), τx=m.draw(:τx, ;αx,ζx), β=m.draw(:β,; μβ,Λβ), ψ=m.draw(:ψ; αψ,ζψ))
    simi = m.updateθpx(dgpi)
    lqxθsimi = -0.5*(x-simi.μx)'*simi.Λx*(x-simi.μx) + simi.μx'*simi.Λx*simi.μx/2
    lqxθsim += lqxθsimi
    Φi = m.formΦ(dgpi.Θ.ϕ; dims)
    sim[:ΦtΦ] .+= Φi'*Φi
    sim[:Φ] .+= Φi
    sim[:τx] += dgpi.Θ.τx
    sim[:τy] += dgpi.Θ.τy
    sim[:β] += dgpi.Θ.β
    sim[:ψ] .+= dgpi.Θ.ψ
  end

  lqxθsim /= iter


  @info "*****lqxθ: $lqxθ; lqxθsim: $lqxθsim"
  #=println("moments:
    sim.ΦtΦ = $(sim[:ΦtΦ] / iter) (act=$EΦtΦ ) 
    sim.Φ = $(sim[:Φ] / iter) (act=$EΦ ) 
    sim.τx  = $(sim[:τx] / iter) (act=$Eτx  )  
    sim.τy = $(sim[:τy] / iter) (act=$Eτy ) 
    sim.β  = $(sim[:β] / iter) (act=$μβ  )  
    sim.ψ  = $(sim[:ψ] / iter) (act=$(EΨ.diag)  )  
  
  ")=#
  #Now test the moments
  #TODO- simulate and finish

end

function testpostτy(dgp; iter)
  @info "testing τy (iter=$iter)"
  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack S,P,T,K = dims
  @unpack αy0, ζy0, M0, ϕ0, β0, βΔ0,  A0, v, αϕ0, ζϕ0, αβ0, ζβ0, ϕmin, ϕmax  = hyper
  @unpack x, ϕ, β, τy, τx, ψ, γ, ν, τβ, τϕ = Θ
  @unpack y,F,r = data

  #start by checking conditional on the moments
  X̃L = m.formX̃L(;x, dims)
  post = lpdist(dgp,:post)
  Ψ = ψ |> m.formΨ
  D = m.formD(γ,v)
  Dinv = D |> inv
  ỹ = m.formỹ(y,x;dims)
  β̃ = m.formβ̃(β, β0)

  Φ= m.formΦ(ϕ; dims)
  @assert (ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ) ≈ (y-Φ*x)'*(y-Φ*x)


  #verify the prior and likelihood
  @assert lpdist(dgp,:τyprior) ≈ (αy0*log(ζy0)-loggamma(αy0) + (αy0-1)*log(τy)-ζy0*τy)
  @assert lpdist(dgp,:ϕprior) ≈ (-τy/2*τϕ *(ϕ-ϕ0)'*M0*(ϕ-ϕ0) + 0.5*(P*log(τy)-P*log(2π)+log(det(M0))+P*log(τϕ)) ) 
  @assert lpdist(dgp,:βprior) ≈ (-(τy*τx/2*τβ)*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0)
    + 0.5*(K*log(τy*τx*τβ)-K*log(2π)+log(det(D*A0*D))))
  @assert lpdist(dgp,:xprior) ≈ (-(τy*τx/2/T)*(sqrt(T)*(x-r)-F*β)'*Ψ*(sqrt(T)*(x-r)-F*β) + 0.5*(T*log(τy*τx) - T*log(T*2π)+log(det(Ψ))))
  @assert lpdist(dgp,:ylike) ≈ (-(τy/2)*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ) + 0.5*(S*log(τy) - S*log(2π)))
  @assert lpdist(dgp,:τϕprior) ≈ (αϕ0*log(ζϕ0)-loggamma(αϕ0) + (αϕ0-1)*log(τϕ)-ζϕ0*τϕ)
  @assert lpdist(dgp,:τβprior) ≈ (αβ0*log(ζβ0)-loggamma(αβ0) + (αβ0-1)*log(τβ)-ζβ0*τβ )    

  lqτyMc1 = (-τy/2*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ)
    -τy/2*τϕ*(ϕ-ϕ0)'*M0*(ϕ-ϕ0)
    -τy*τx/2/T*(sqrt(T)*(x-r)-F*β)'*Ψ*(sqrt(T)*(x-r)-F*β)
    -τy*τx/2*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0)
    -ζy0*τy+((S+T+K+P)/2+αy0-1)*log(τy))
  c1=(-(S+T+P+K)/2*log(2π) + 0.5*log(det(Ψ)) + 0.5*log(det(M0)) + 0.5*P*log(τϕ) + 0.5*K*log(τβ)
    +0.5*log(det(D*A0*D)) + (T+K)/2*log(τx) + αy0*log(ζy0) - loggamma(αy0)-T/2*log(T)
    +sum([:γprior,:ωprior,:νprior,:ψprior,:τxprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p)))  
  @assert lqτyMc1+c1 ≈ post "lqτyMc1+c1: $(lqτyMc1+c1); post: $post"

  #conditional updating equations
  αy=(S+T+K+P)/2+αy0
  ζy = (0.5*(ỹ'ỹ + ϕ'*X̃L'*X̃L*ϕ - ỹ'*X̃L*ϕ - ϕ'*X̃L'*ỹ)
    +0.5*τϕ*(ϕ'*M0*ϕ + ϕ0'*M0*ϕ0 - ϕ0'*M0*ϕ - ϕ'*M0*ϕ0)
    +0.5*τx*(x'*Ψ*x + (β'*F'+sqrt(T)*r')*Ψ*(F*β+sqrt(T)*r)/T - x'*Ψ*(F*β+sqrt(T)*r)/sqrt(T) - (β'*F'+sqrt(T)*r')*Ψ*x/sqrt(T))
    +0.5*τx*τβ*((β-β0)'*D*A0*D*(β-β0) + βΔ0'*A0*βΔ0 - βΔ0'*A0*D*(β-β0) - (β-β0)'*D*A0*βΔ0) + ζy0)
  @assert αy ≈ m.updateθpτy(dgp).αy
  @assert ζy ≈ m.updateθpτy(dgp).ζy "ζy: $ζy; m.updateθpτy(dgp).ζy: $(m.updateθpτy(dgp).ζy)"

  Random.seed!(11)
  manualdraw = m.draw(:τy; αy, ζy)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:τy; dgp).Θ.τy
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

  lqτyMc2 = logpdf(Gamma(αy, 1/ζy), τy)
  c2 = c1 - αy*log(ζy) + loggamma(αy)
  @assert lqτyMc2 + c2 ≈ post "lqτyMc2+c2: $(lqτyMc2 + c2); post: $post" 

  #Now need to check the expectations
  #need placeholders for the updates
  μx=x.-0.1
  Λx=τy*τx .* m.formΨ(ψ)
  αx=hyper.αx0 + 0.4
  ζx=hyper.ζx0 + 0.3
  αϕ=hyper.αϕ0-0.2
  ζϕ=hyper.ζϕ0-0.1
  αβ=hyper.αβ0-0.4
  ζβ=hyper.ζβ0-0.3

  αψ=[ν/2 for i ∈ 1:T]
  ζψ=[ν/2 for i ∈ 1:T]
  D = m.formD(Θ.γ,hyper.v)
  pγ = 1.0 .- exp.(-0.1 .* collect(1:K))
  μβ= D*β
  Λβ = (τx*τy .* D*hyper.A0*D)*10
  μϕ=ϕ
  Λϕ=τy .* M0

  vbϕ = m.VBqϕ(dgp; μϕ, Λϕ, testmoments=true)
  @unpack EΦ,EΦtΦ, EϕtM0ϕ, Σϕ = vbϕ
  vbx = m.VBqx(dgp; μx, Λx, testmoments=true)
  @unpack EXL, EXLtXL, EX̃LtX̃L, EX̃L, Eỹ, EX̃Ltỹ, Eỹtỹ, Σx = vbx
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx
  vbτϕ = m.VBqτϕ(dgp; αϕ, ζϕ, testmoments=true)
  @unpack Eτϕ= vbτϕ
  vbτβ = m.VBqτβ(dgp; αβ, ζβ, testmoments=true)
  @unpack Eτβ= vbτβ    
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack μβ, Σβ, Eβ̃= vbβ
  vbγ = m.VBqγ(dgp; pγ, testmoments=true)
  @unpack pγ, ED, EDA0D= vbγ
  vbψ = m.VBqψ(dgp;  αψ, ζψ, testmoments=true)
  @unpack EΨ= vbψ  
 


  #compute and test the necessary moments
  ExtΨx = m.moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrT12tΨFβPrT12 = m.moment(:EβtFtPrT12tΨFβPrT12; μβ, Σβ, F, EΨ, r, dims)
  EϕtX̃LtX̃Lϕ = m.moment(:EϕtX̃LtX̃Lϕ; μϕ, Σϕ, EX̃LtX̃L,dims)
  #EϕtM0ϕ = m.moment(:EϕtM0ϕ; μϕ, Σϕ, M0)
  Eβ̃tDA0Dβ̃ = m.moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D, )
  EβtDA0Dβ = m.moment(:EβtDA0Dβ; μβ, Σβ, EDA0D,)

  m.testmoment(:ExtΨx; μx, Λx, EΨ)
  m.testmoment(:EβtFtPrT12tΨFβPrT12; μβ, Λβ, F, EΨ, r, dims)
  m.testmoment(:EϕtX̃LtX̃Lϕ; μϕ, Λϕ, μx, Λx, dims)
  m.testmoment(:EβtDA0Dβ; μβ, Λβ, EDA0D,)
  m.testmoment(:Eβ̃tDA0Dβ̃; μβ, Λβ, EDA0D, β0)


  dgpmom = m.DGP(dgp; 
    x=vbx, 
    τx=vbτx,
    γ=vbγ, 
    ϕ=vbϕ, 
    β=vbβ,
    τϕ=vbτϕ,
    τβ=vbτβ, 
    ψ=vbψ,strict=false)


  @unpack αy,ζy = m.updateθqτy(dgpmom)
  @assert αy ≈ (S+T+P+K)/2+αy0
  @assert ζy ≈ (0.5*(Eỹtỹ + EϕtX̃LtX̃Lϕ - EX̃Ltỹ'*μϕ - μϕ'*EX̃Ltỹ)
    + 0.5*Eτϕ*(EϕtM0ϕ + ϕ0'M0*ϕ0 - ϕ0'M0*μϕ - μϕ'*M0*ϕ0)
    + 0.5*Eτx*(ExtΨx + EβtFtPrT12tΨFβPrT12/T - μx'*EΨ*(F*μβ+sqrt(T)*r)/T - (μβ'F'+sqrt(T)*r')*EΨ*μx/T)
    + 0.5*Eτx*Eτβ*(EβtDA0Dβ - 2*β0'*EDA0D*μβ + β0'EDA0D*β0
      + βΔ0'A0*βΔ0 - βΔ0'A0*ED*(μβ-β0) - (μβ-β0)'ED*A0*βΔ0) + ζy0)
  lqτyθ = (αy-1.0)*log(τy) .- ζy*τy 


  lqτyθsim = 0.0
  sim = Dict(:xtΨx=>0.0, :βtFtPrT12tΨFβPrT12=>0.0, :ϕtX̃LtX̃Lϕ=>0.0,  :SX̃LtX̃L=>0.0,
    :ϕtM0ϕ=>0.0, :DA0D=>zeros(K,K), :β̃tDA0Dβ̃=>0.0, :ỹtỹ=>0.0,
    :τβ=>0.0, :τϕ=>0.0)

  d = (;μϕ, Λϕ, αx,ζx, μβ,Λβ, μx, Λx, EXL)
  @eval Main d=$d

  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      ϕ=m.draw(:ϕ;μϕ, Λϕ, Σϕ=m.pdinv(Λϕ), ϕmin, ϕmax), 
      τx=m.draw(:τx,; αx,ζx),
      τϕ=m.draw(:τϕ,; αϕ,ζϕ), 
      τβ =m.draw(:τβ,; αβ ,ζβ ),  
      γ=m.draw(:γ; pγ),
      β=m.draw(:β,; μβ,Λβ), 
      ψ=m.draw(:ψ; αψ,ζψ), 
      x=m.draw(:x; μx,Λx),)
    simi = m.updateθpτy(dgpi)
    lqτyθsimi = (simi.αy-1)*log(τy)-simi.ζy*τy
    lqτyθsim += lqτyθsimi

    Θi = dgpi.Θ
    simD = m.formD(Θi.γ,v)
    simΨ = m.formΨ(Θi.ψ)
    simX̃L = m.formX̃L(;x=Θi.x, dims)
    simỹ = m.formỹ(;y=data.y, x=Θi.x, dims)

    sim[:βtFtPrT12tΨFβPrT12] += (Θi.β'*F'+r'*sqrt(T))*simΨ*(F*Θi.β+r*sqrt(T))
    sim[:xtΨx] += Θi.x'*simΨ*Θi.x
    sim[:ϕtX̃LtX̃Lϕ] += Θi.ϕ'*simX̃L'*simX̃L*Θi.ϕ
    sim[:SX̃LtX̃L] += sum(simX̃L'*simX̃L)
    sim[:ϕtM0ϕ] += Θi.ϕ'*M0*Θi.ϕ
    sim[:DA0D] .+= simD*A0*simD
    sim[:β̃tDA0Dβ̃] += (Θi.β-β0)'*simD*A0*simD*(Θi.β-β0)
    sim[:ỹtỹ] += simỹ'simỹ
    sim[:τβ] += Θi.τβ
    sim[:τϕ] += Θi.τϕ


  end

  lqτyθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  println("xtΨx: sim $(sim[:xtΨx]); act $(ExtΨx) ")
  println("βtFtPrT12tΨFβPrT12: sim $(sim[:βtFtPrT12tΨFβPrT12]); act $(EβtFtPrT12tΨFβPrT12) ")
  println("ϕtX̃LtX̃Lϕ: sim $(sim[:ϕtX̃LtX̃Lϕ]); act $(EϕtX̃LtX̃Lϕ) ")  
  println("SX̃LtX̃L: sim $(sim[:SX̃LtX̃L]); act $(EX̃LtX̃L |> sum) ") 
  println("ϕtM0ϕ: sim $(sim[:ϕtM0ϕ]); act $(EϕtM0ϕ) ")
  println("DA0D: sim $(sim[:DA0D]); \nact $(EDA0D) ")  
  println("β̃tDA0Dβ̃: sim $(sim[:β̃tDA0Dβ̃]); act $(Eβ̃tDA0Dβ̃ |> sum) ") 
  println("ỹtỹ: sim $(sim[:ỹtỹ]); act $(Eỹtỹ) ") 
  println("τϕ: sim $(sim[:τϕ]); act $(Eτϕ) ") 
  println("τβ: sim $(sim[:τβ]); act $(Eτβ) ") 

  @info "*****lqτyθ: $lqτyθ; lqτyθsim: $lqτyθsim"





end


function testpostτx(dgp; iter)
  @info "testing τx (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack αx0, ζx0, β0, βΔ0, A0, v, αβ0, ζβ0 = hyper
  @unpack x, β, τy, τx, ψ, γ, ν, τβ = Θ
  @unpack F,r = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  Ψ = ψ |> m.formΨ
  D = m.formD(γ,v)
  Dinv = D |> inv
  β̃ = m.formβ̃(β,β0)

  #Φ= m.formΦ(ϕ; dims)
  #@assert (ỹ-XL*ϕ)'*(ỹ-XL*ϕ) ≈ (y-Φ*x)'*(y-Φ*x)


  #verify the prior and likelihood
  @assert lpdist(dgp,:τxprior) ≈ (αx0*log(ζx0)-loggamma(αx0) + (αx0-1)*log(τx)-ζx0*τx)
  @assert lpdist(dgp,:βprior) ≈ (-(τy*τx/2*τβ)*(β̃-Dinv*βΔ0)'*D*A0*D*(β̃-Dinv*βΔ0)
    + 0.5*(K*log(τy*τx)-K*log(2π)+log(det(D*A0*D))+K*log(τβ)))
  @assert lpdist(dgp,:xprior) ≈ (-(τy*τx/2)*(sqrt(T)*(x-r)-F*β)'*Ψ*(sqrt(T)*(x-r)-F*β)/T + 0.5*(T*log(τy*τx/T) - T*log(2π)+log(det(Ψ))))
  @assert lpdist(dgp,:τβprior) ≈ (αβ0*log(ζβ0)-loggamma(αβ0) + (αβ0-1)*log(τβ)-ζβ0*τβ)  


  lqτxMc1 = (-τy*τx/2*(sqrt(T)*(x-r)-F*β)'*Ψ*(sqrt(T)*(x-r)-F*β)/T
    -τy*τx/2*τβ*(β̃-Dinv*βΔ0)'*D*A0*D*(β̃-Dinv*βΔ0)
    -τx*ζx0+((T+K)/2+αx0-1)*log(τx))
  c1 = (-(T+K)/2*log(2π)+0.5*log(det(Ψ)) +0.5*log(det(D*A0*D))-T/2*log(T)
    +(T+K)/2*log(τy)+αx0*log(ζx0)-loggamma(αx0)+0.5*K*log(τβ)
    +sum([:ylike, :ϕprior, :γprior,:ωprior,:ψprior,:τyprior,:νprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p))) 
  @assert lqτxMc1+c1 ≈ post "lqτxMc1+c1: $(lqτxMc1+c1); post: $post"

  #conditional updating equations
  αx=(T+K)/2+αx0
  ζx = (ζx0 + 0.5*τy*(x'*Ψ*x + (β'*F'+sqrt(T)*r')*Ψ*(F*β +sqrt(T)*r)/T
    - x'*Ψ*(F*β +sqrt(T)*r)/sqrt(T)- (β'*F'+sqrt(T)*r')*Ψ*x/sqrt(T))
    +0.5*τy*τβ*(β̃'*D*A0*D*β̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*D*β̃ - β̃'*D*A0*βΔ0))
  @assert αx ≈ m.updateθpτx(dgp).αx
  @assert ζx ≈ m.updateθpτx(dgp).ζx "ζx: $ζx; m.updateθpτx(dgp).ζx: $(m.updateθpτx(dgp).ζx)"

  Random.seed!(11)
  manualdraw = m.draw(:τx; αx, ζx)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:τx; dgp).Θ.τx
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

  lqτxMc2 = logpdf(Gamma(αx, 1/ζx), τx)
  c2 = c1 - αx*log(ζx) + loggamma(αx)
  @assert lqτxMc2 + c2 ≈ post "lqτxMc2+c2: $(lqτxMc2 + c2); post: $post" 

  
  #Now need to check the expectations
  #need placeholders for the updates
  μx=x.-0.1
  Λx=τy*τx .* m.formΨ(ψ)
  αy=hyper.αy0 + 0.4
  ζy=hyper.ζy0 + 0.3
  αβ=hyper.αβ0-0.4
  ζβ=hyper.ζβ0-0.3
  αψ=[ν/2 for i ∈ 1:T]
  ζψ=[ν/2 for i ∈ 1:T]
  D = m.formD(Θ.γ,hyper.v)
  pγ = 1.0 .- exp.(-0.1 .* collect(1:K))
  μβ= D*β
  Λβ = (τx*τy .* D*hyper.A0*D)*10
  Σβ = m.pdinv(Λβ)

  vbx = m.VBqx(dgp; μx, Λx, testmoments=true)
  @unpack Σx = vbx
  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτβ  = m.VBqτβ(dgp; αβ, ζβ, testmoments=true)
  @unpack Eτβ= vbτβ 
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack μβ, Σβ, Eβ̃= vbβ 
  vbγ = m.VBqγ(dgp; pγ, testmoments=true)
  @unpack pγ, ED, EDA0D= vbγ
  vbψ = m.VBqψ(dgp;  αψ, ζψ, testmoments=true)
  @unpack EΨ= vbψ  

  #compute and test the necessary moments
  ExtΨx = m.moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrT12tΨFβPrT12 = m.moment(:EβtFtPrT12tΨFβPrT12; μβ, Σβ, F, EΨ,r,dims)
  ED = m.moment(:ED; pγ, v)  
  EDA0D = m.moment(:EDA0D; pγ, A0, v, ED)
  Eβ̃tDA0Dβ̃ = m.moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)

  m.testmoment(:EβtDA0Dβ; μβ, Λβ, EDA0D,)
  m.testmoment(:Eβ̃tDA0Dβ̃; μβ, Λβ, EDA0D, β0)
  m.testmoment(:ExtΨx; μx, Λx, EΨ)
  m.testmoment(:EβtFtPrT12tΨFβPrT12; μβ, Λβ, F, EΨ,r,dims)


  dgpmom = m.DGP(dgp; x=vbx, τy=vbτy,γ=vbγ, 
    β=vbβ, ψ=vbψ,τβ=vbτβ, strict=false)

  @unpack αx,ζx = m.updateθqτx(dgpmom)
  @assert αx ≈(T+K)/2+αx0
  @assert ζx ≈ (ζx0 + 0.5*Eτy*(ExtΨx+EβtFtPrT12tΨFβPrT12/T
    -μx'EΨ*(F*μβ+r*sqrt(T))/sqrt(T)-(μβ'*F'+sqrt(T)*r')*EΨ*μx/sqrt(T)) 
    +0.5*Eτy*Eτβ*(Eβ̃tDA0Dβ̃  + βΔ0'A0*βΔ0 - βΔ0'A0*ED*(μβ-β0) -  (μβ-β0)'ED*A0*βΔ0))
  lqτxθ = (αx - 1.0)*log(τx) .- ζx*τx 


  lqτxθsim = 0.0
  sim = Dict(:xtΨx=>0.0, :βtFtPrT12tΨFβPrT12=>0.0, 
    :DA0D=>zeros(K,K), :β̃tDA0Dβ̃=>0.0,
    :αx=>0.0, :ζx=>0.0, :τβ=>0.0)

  d = (;αx,ζx, μβ,Λβ, μx, Λx,)
  @eval Main d=$d

  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τy=m.draw(:τy, ;αy,ζy), 
      τβ=m.draw(:τβ, ;αβ,ζβ),      
      γ=m.draw(:γ; pγ),
      β=m.draw(:β,; μβ,Λβ), 
      ψ=m.draw(:ψ; αψ,ζψ), 
      x=m.draw(:x;μx, Σx),)
    simi = m.updateθpτx(dgpi)
    lqτxθsimi = (simi.αx-1)*log(τx)-simi.ζx*τx
    lqτxθsim += lqτxθsimi

    Θi = dgpi.Θ
    simD = m.formD(Θi.γ,v)
    simΨ = m.formΨ(Θi.ψ)
    simβ̃=m.formβ̃(Θi.β, β0)   
    sim[:βtFtPrT12tΨFβPrT12] += (Θi.β'*F'+sqrt(T)*r')*simΨ*(F*Θi.β+sqrt(T)*r)
    sim[:xtΨx] += Θi.x'*simΨ*Θi.x
    sim[:DA0D] .+= (simD*A0*simD)
    sim[:β̃tDA0Dβ̃] += simβ̃'*simD*A0*simD*simβ̃
    sim[:αx] += simi.αx
    sim[:ζx] += simi.ζx
    sim[:τβ] += Θi.τβ


  end

  lqτxθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  @eval Main sim=$sim

  println("xtΨx: sim $(sim[:xtΨx]); act $(ExtΨx) ")
  println("βtFtPrT12tΨFβPrT12: sim $(sim[:βtFtPrT12tΨFβPrT12]); act $(EβtFtPrT12tΨFβPrT12) ")
  println("DA0D: sim $(sim[:DA0D]); \nact $(EDA0D) ")  
  println("β̃tDA0Dβ̃: sim $(sim[:β̃tDA0Dβ̃]); act $(Eβ̃tDA0Dβ̃ |> sum) ") 
  println("τβ: sim $(sim[:τβ]); \nact $(Eτβ) ") 

  @info "*****lqτxθ: $lqτxθ; lqτxθsim: $lqτxθsim"

end


function testpostτϕ(dgp; iter)
  @info "testing τϕ (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack P = dims
  @unpack αy0, ζy0, M0, ϕ0, αϕ0, ζϕ0, ϕmin, ϕmax = hyper
  @unpack ϕ, τϕ, τy = Θ
  @unpack F,r = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)



  #verify the prior and likelihood
  @assert lpdist(dgp,:τyprior) ≈ (αy0*log(ζy0)-loggamma(αy0) + (αy0-1)*log(τy)-ζy0*τy)
  @assert lpdist(dgp,:ϕprior) ≈ (-(τy/2*τϕ)*(ϕ - ϕ0)'*M0*(ϕ - ϕ0)
    + 0.5*(P*log(τy)-P*log(2π)+log(det(M0))+P*log(τϕ)))
  @assert lpdist(dgp,:τϕprior) ≈ (αϕ0*log(ζϕ0)-loggamma(αϕ0) + (αϕ0-1)*log(τϕ)-ζϕ0*τϕ)  



  lqτϕMc1 = (-τy/2*τϕ*(ϕ - ϕ0)'*M0*(ϕ - ϕ0)
    -ζϕ0*τϕ+(αϕ0-1+P/2)*log(τϕ))
  c1 = (P/2*log(τy/ (2π))+0.5*log(det(M0))
    +αϕ0*log(ζϕ0)-loggamma(αϕ0)
    +sum([:ylike, :βprior, :γprior,:ωprior,:ψprior,:τyprior,:νprior, :xprior, :τxprior, :τβprior] .|> p->lpdist(dgp, p))) 
  @assert lqτϕMc1+c1 ≈ post "lqτϕMc1+c1: $(lqτϕMc1+c1); post: $post"



  #conditional updating equations
  αϕ=P/2+αϕ0
  ζϕ = ζϕ0 + τy/2*(ϕ - ϕ0)'*M0*(ϕ - ϕ0)
  #(0.5*τx*τx*(β-Dinv*β0)'*D*A0*D*(β-Dinv*β0) + ζgx0)
  @assert αϕ ≈ m.updateθpτϕ(dgp).αϕ
  @assert ζϕ ≈ m.updateθpτϕ(dgp).ζϕ "ζτϕ: $ζτϕ; m.updateθpτϕ(dgp).ζτϕ: $(m.updateθpτϕ(dgp).ζτϕ)"
  


  Random.seed!(11)
  manualdraw = m.draw(:τϕ; αϕ, ζϕ)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:τϕ; dgp).Θ.τϕ
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

  lqτϕMc2 = logpdf(Gamma(αϕ, 1/ζϕ), τϕ)
  c2 = c1 - αϕ*log(ζϕ) + loggamma(αϕ)
  @assert lqτϕMc2 + c2 ≈ post "lqτϕMc2+c2: $(lqτϕMc2 + c2); post: $post" 


  #Now need to check the expectations
  #need placeholders for the updates
  αy=hyper.αy0 + 0.4
  ζy=hyper.ζy0 + 0.3
  μϕ=ϕ
  Λϕ=τy .* M0


  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbϕ = m.VBqϕ(dgp; μϕ, Λϕ, testmoments=true)
  @unpack μϕ, Λϕ, EΦ,EΦtΦ,EϕtM0ϕ = vbϕ


  dgpmom = m.DGP(dgp; τy=vbτy,γ=vbϕ, strict=false)


  @unpack αϕ,ζϕ = m.updateθqτϕ(dgpmom)
  @assert αϕ ≈ P/2+αϕ0
  @assert ζϕ ≈ (ζϕ0 + 0.5*Eτy*(EϕtM0ϕ + ϕ0'M0*ϕ0 - ϕ0'M0*μϕ - μϕ'*M0*ϕ0))
  lqτϕθ = (αϕ - 1.0)*log(τϕ) .- ζϕ*τϕ 
  


  lqτϕθsim = 0.0
  sim = Dict(:ϕtM0ϕ=>0.0, :αϕ=>0.0, :ζϕ=>0.0, )

  #d = (;αx,ζx, μβ,Λβ, )
  #@eval Main d=$d


  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τy=m.draw(:τy, ;αy,ζy),       
      ϕ=m.draw(:ϕ;μϕ, Λϕ, Σϕ=m.pdinv(Λϕ), ϕmin, ϕmax), 
      )
    
      simi = m.updateθpτϕ(dgpi)
    lqτϕθsimi = (simi.αϕ-1)*log(τϕ)-simi.ζϕ*τϕ
    lqτϕθsim += lqτϕθsimi

    Θi = dgpi.Θ
    sim[:ϕtM0ϕ] += (Θi.ϕ'*M0*Θi.ϕ)
    sim[:αϕ] += simi.αϕ
    sim[:ζϕ] += simi.ζϕ


  end

  lqτϕθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  @eval Main sim=$sim


  println("ϕtM0ϕ: sim $(sim[:ϕtM0ϕ]); \nact $(EϕtM0ϕ) ")  
  @info "*****lqτϕθ: $lqτϕθ; lqτβθsim: $lqτϕθsim"
end

function testpostτβ(dgp; iter)
  @info "testing τβ (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack αx0, ζx0, β0, βΔ0, A0, v, αβ0, ζβ0 = hyper
  @unpack β, τy, τx, ψ, γ, ν, τβ = Θ
  @unpack F,r = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  D = m.formD(γ,v)
  Dinv = D |> inv
  β̃ = m.formβ̃(β,β0)

  #Φ= m.formΦ(ϕ; dims)
  #@assert (ỹ-XL*ϕ)'*(ỹ-XL*ϕ) ≈ (y-Φ*x)'*(y-Φ*x)


  #verify the prior and likelihood
  @assert lpdist(dgp,:τxprior) ≈ (αx0*log(ζx0)-loggamma(αx0) + (αx0-1)*log(τx)-ζx0*τx)
  @assert lpdist(dgp,:βprior) ≈ (-(τy*τx/2*τβ)*(β̃-Dinv*βΔ0)'*D*A0*D*(β̃-Dinv*βΔ0)
    + 0.5*(K*log(τy*τx)-K*log(2π)+log(det(D*A0*D))+K*log(τβ)))
  @assert lpdist(dgp,:τβprior) ≈ (αβ0*log(ζβ0)-loggamma(αβ0) + (αβ0-1)*log(τβ)-ζβ0*τβ)  



  lqτβMc1 = (-τy*τx/2*τβ*(β̃-Dinv*βΔ0)'*D*A0*D*(β̃-Dinv*βΔ0)
    -ζβ0*τβ+(αβ0-1+K/2)*log(τβ))
  c1 = (K/2*log((τx*τy/ 2π))+0.5*log(det(D*A0*D))
    +αβ0*log(ζβ0)-loggamma(αβ0)
    +sum([:ylike, :ϕprior, :γprior,:ωprior,:ψprior,:τyprior,:νprior, :xprior, :τxprior, :τϕprior] .|> p->lpdist(dgp, p))) 
  @assert lqτβMc1+c1 ≈ post "lqτβMc1+c1: $(lqτβMc1+c1); post: $post"


  #conditional updating equations
  αβ=K/2+αβ0
  ζβ = ζβ0 + 0.5*τy*τx*(β̃'*D*A0*D*β̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*D*β̃ - β̃'*D*A0*βΔ0)
  #(0.5*τx*τx*(β-Dinv*β0)'*D*A0*D*(β-Dinv*β0) + ζτβ0)
  @assert αβ ≈ m.updateθpτβ(dgp).αβ
  @assert ζβ ≈ m.updateθpτβ(dgp).ζβ "ζβ: $ζβ; m.updateθpτβ(dgp).ζgβ: $(m.updateθpτβ(dgp).ζβ)"
  

  Random.seed!(11)
  manualdraw = m.draw(:τβ; αβ, ζβ)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:τβ; dgp).Θ.τβ
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

  lqτβMc2 = logpdf(Gamma(αβ, 1/ζβ), τβ)
  c2 = c1 - αβ*log(ζβ) + loggamma(αβ)
  @assert lqτβMc2 + c2 ≈ post "lqτβMc2+c2: $(lqτβMc2 + c2); post: $post" 


  #Now need to check the expectations
  #need placeholders for the updates
  αy=hyper.αy0 + 0.4
  ζy=hyper.ζy0 + 0.3
  αx=hyper.αx0 + 0.2
  ζx=hyper.ζx0 + 0.1  
  D = m.formD(Θ.γ,hyper.v)
  pγ = 1.0 .- exp.(-0.1 .* collect(1:K))
  μβ= D*β
  Λβ = (τx*τy .* D*hyper.A0*D)*10
  Σβ = m.pdinv(Λβ)

  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx  
  vbτβ = m.VBqτβ(dgp; αβ, ζβ, testmoments=true)
  @unpack Eτβ = vbτβ  
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack μβ, Σβ, Eβ̃= vbβ 
  vbγ = m.VBqγ(dgp; pγ, testmoments=true)
  @unpack pγ, ED, EDA0D= vbγ

  #compute and test the necessary moments
  ED = m.moment(:ED; pγ, v)  
  EDA0D = m.moment(:EDA0D; pγ, A0, v, ED)
  Eβ̃tDA0Dβ̃ = m.moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)
  
  m.testmoment(:Eβ̃tDA0Dβ̃; Λβ, EDA0D,μβ,β0)

  dgpmom = m.DGP(dgp; τy=vbτy,γ=vbγ, 
    β=vbβ, τx=vbτx, strict=false)



  @unpack αβ,ζβ = m.updateθqτβ(dgpmom)
  @assert αβ ≈ K/2+αβ0
  @assert ζβ ≈ (ζβ0 + 0.5*Eτy*Eτx*(Eβ̃tDA0Dβ̃ + βΔ0'A0*βΔ0 - βΔ0'A0*ED*(μβ-β0) -  (μβ-β0)'ED*A0*βΔ0))
  lqτβθ = (αβ - 1.0)*log(τβ) .- ζβ*τβ 


  lqτβθsim = 0.0
  sim = Dict(:DA0D=>zeros(K,K), :β̃tDA0Dβ̃=>0.0,
    :αβ=>0.0, :ζβ=>0.0, )

  d = (;αx,ζx, μβ,Λβ, )
  @eval Main d=$d


  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τy=m.draw(:τy, ;αy,ζy), 
      τx=m.draw(:τx, ;αx,ζx),      
      γ=m.draw(:γ; pγ),
      β=m.draw(:β,; μβ,Λβ))
    simi = m.updateθpτβ(dgpi)
    lqτβθsimi = (simi.αβ-1)*log(τβ)-simi.ζβ*τβ
    lqτβθsim += lqτβθsimi

    Θi = dgpi.Θ
    simD = m.formD(Θi.γ,v)
    simβ̃ = m.formβ̃(Θi.β,β0)
    sim[:DA0D] .+= (simD*A0*simD)
    sim[:β̃tDA0Dβ̃] += simβ̃'*simD*A0*simD*simβ̃ 
    sim[:αβ] += simi.αβ
    sim[:ζβ] += simi.ζβ


  end

  lqτβθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  @eval Main sim=$sim


  println("DA0D: sim $(sim[:DA0D]); \nact $(EDA0D) ")  
  println("β̃tDA0Dβ̃: sim $(sim[:β̃tDA0Dβ̃]); act $(Eβ̃tDA0Dβ̃ |> sum) ") 
  @info "*****lqτβθ: $lqτβθ; lqτβθsim: $lqτβθsim"
end

function testpostβ(dgp; iter)
  @info "testing β (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack β0, βΔ0, A0, v = hyper
  @unpack x, β, τy, τx, ψ, γ, ν, τβ = Θ
  @unpack F,r = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  Ψ = ψ |> m.formΨ
  D = m.formD(γ,v)
  Dinv = D |> inv

  #verify the prior and likelihood
  @assert lpdist(dgp,:βprior) ≈ (-(τy*τx/2*τβ)*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0)
    + 0.5*(K*log(τy*τx*τβ)-K*log(2π)+log(det(D*A0*D))))
  @assert lpdist(dgp,:xprior) ≈ (-(τy*τx/2/T)*((x-r)*sqrt(T)-F*β)'*Ψ*((x-r)*sqrt(T)-F*β) 
    + 0.5*(T*log(τy*τx/T) - T*log(2π)+log(det(Ψ))))

  lqβMc1 = (-τy*τx/2*((x-r)*sqrt(T)-F*β)'*Ψ*((x-r)*sqrt(T)-F*β)/T 
    - τy*τx/2*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0))
  c1 = ((T+K)/2*log(τx*τy/(2π))+0.5*log(det(Ψ)) +0.5*log(det(D*A0*D)) + 0.5*K*log(τβ) - 0.5*T*log(T)
    +sum([:ylike, :ϕprior, :γprior,:ωprior,:ψprior,:νprior,:τyprior,:τxprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p))) 
  @assert lqβMc1+c1 ≈ post "lqβMc1+c1: $(lqβMc1+c1); post: $post"

  lqβMc2 = -τy*τx/2*(β'F'Ψ*F*β/T - (x-r)'Ψ*F*β/sqrt(T) - β'F'Ψ*(x-r)/sqrt(T)+ (τβ.* (β'D*A0*D*β - 2*β'D*A0*(D*β0+βΔ0))))
  c2 = c1 - τy*τx/2*(τβ*(D*β0+βΔ0)'*A0*(D*β0+βΔ0)+(x-r)'*Ψ*(x-r))
  @assert lqβMc2 + c2 ≈ post "lqβMc2+c2: $(lqβMc2 + c2); post: $post" 

  #conditional updating equations
  Λβ=τx*τy*(F'*Ψ*F/T + (D*A0*D.*τβ))
  μβ=τx*τy*inv(Λβ)*(F'*Ψ*(x-r)/sqrt(T) + ( τβ .*D*A0*(D*β0+βΔ0)))
  @assert Λβ ≈ m.updateθpβ(dgp).Λβ
  @assert μβ ≈ m.updateθpβ(dgp).μβ

  #test the conditional draw
  Random.seed!(11)
  manualdraw = m.draw(:β; μβ, Σβ=m.pdinv(Λβ))
  Random.seed!(11)
  conddraw = m.conditionaldraw(:β; dgp).Θ.β
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"


  lqβMc3 = -0.5*(β-μβ)'Λβ*(β-μβ)
  c3 = c2 + 0.5*μβ'*Λβ*μβ
  @assert lqβMc3 + c3 ≈ post "lqβMc3+c3: $(lqβMc3 + c3); post: $post" 

  lqβMc4 = logpdf(MultivariateNormal(μβ, m.pdinv(Λβ)), β)
  c4 = c3 + K/2*log(2π) - 0.5*logdet(Λβ)
  @assert lqβMc4 + c4 ≈ post "lqβMc4+c4: $(lqβMc4 + c4); post: $post" 
  
  #Now need to check the expectations
  #need placeholders for the updates
  μx=x.-0.1
  Λx=τy*τx .* m.formΨ(ψ)
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  αx=hyper.αx0+0.4
  ζx=hyper.ζx0+0.3
  αβ=hyper.αβ0-0.4
  ζβ=hyper.ζβ0-0.3  
  αψ=[ν/2 for i ∈ 1:T]
  ζψ=[ν/2 for i ∈ 1:T]
  D = m.formD(Θ.γ,hyper.v)
  pγ = 1.0 .- exp.(-0.1 .* collect(1:K))
  μβ= D*β
  Λβ = (τx*τy .* D*hyper.A0*D)*10

  vbx = m.VBqx(dgp; μx, Λx, testmoments=true)
  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx
  vbτβ = m.VBqτβ(dgp; αβ, ζβ, testmoments=true)
  @unpack Eτβ= vbτβ 
  vbγ = m.VBqγ(dgp; pγ, testmoments=true)
  @unpack ED, EDA0D= vbγ
  vbψ = m.VBqψ(dgp;  αψ, ζψ, testmoments=true)
  @unpack EΨ= vbψ  

  #compute and test the necessary moments
  m.testmoment(:ED; pγ, v)  
  m.testmoment(:EDA0D; pγ, A0, v)

  dgpmom = m.DGP(dgp; γ=vbγ, x=vbx,τβ=vbτβ,
    ψ=vbψ, τx=vbτx, τy=vbτy,strict=false)

  #compute and verify q(β)
  @unpack μβ,Λβ = m.updateθqβ(dgpmom)
  @assert Λβ ≈ Eτx*Eτy*(F'*EΨ*F/T+(EDA0D .*Eτβ))
  @assert μβ ≈ Eτx*Eτy*inv(Λβ)*(F'*EΨ*(μx-r)/sqrt(T)+(Eτβ.*( EDA0D*β0+ED*A0*βΔ0)))

  #note we subtract off the constant to allow for easy comparison
  lqβθ = -0.5*(β-μβ)'*Λβ*(β-μβ)+μβ'*Λβ*μβ/2


  lqβθsim = 0.0
  sim = Dict( :EDA0D=>zeros(K,K),)

  #d = (;αx,ζx, μβ,Λβ, μx, Λx,)
  #@eval Main d=$d
  lqβθsims = Vector{Float64}(undef,iter)
  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τx=m.draw(:τx, ;αx,ζx),     
      τy=m.draw(:τy, ;αy,ζy), 
      τβ=m.draw(:τβ, ;αβ,ζβ),       
      γ=m.draw(:γ; pγ),
      β=m.draw(:β,; μβ,Λβ), 
      ψ=m.draw(:ψ; αψ,ζψ), 
      x=m.draw(:x;μx, Λx),)
    simi = m.updateθpβ(dgpi)
    lqβθsimi = -0.5*(β-simi.μβ)'*simi.Λβ*(β-simi.μβ)+simi.μβ'*simi.Λβ*simi.μβ/2
    lqβθsim += lqβθsimi

    Θi = dgpi.Θ
    simD = m.formD(Θi.γ,v)
    sim[:EDA0D] .+= (simD*A0*simD)


  end

  lqβθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  println("DA0D: sim $(sim[:EDA0D]);\n act $(EDA0D) ")  

  @info "*****lqβθ: $lqβθ; lqβθsim: $lqβθsim"

end

#faster version for diagonal A0
function testpostγ(dgp, A0::Diagonal; iter)
  @info "testing DIAGONAL γ (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack β0, βΔ0, v, κ0, δ0, αβ0, ζβ0 = hyper
  @unpack x, β, τy, τx, γ, ω, τβ = Θ
  @unpack F = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  D = m.formD(γ,v)
  Dinv = D |> inv
  d=D.diag
  a0=A0.diag

  #verify the prior and likelihood
  @assert lpdist(dgp,:βprior) ≈ (-(τy*τx/2*τβ)*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0)
    + 0.5*(K*log(τy*τx)-K*log(2π)+log(det(D*A0*D)) + K*log(τβ)))
  @assert lpdist(dgp,:γprior) ≈ ((γ .|> γk-> log(ω^(γk)*(1-ω)^(1-γk))) |> sum)

  pγ = missings(Float64, K)
  lpγ = similar(pγ)
  lpγc = similar(pγ)
  for k ∈ 1:K
    Mk = Not(k)
    lqγkMc1 = log(d[k])-τx*τy*d[k]^2*τβ*a0[k]/2*(β[k]-β0[k]-βΔ0[k]/d[k])^2+γ[k]*log(ω)+(1-γ[k])*log(1-ω)
    c1 = (0.5*log(τx*τy*τβ *a0[k]/(2π)) + logpdf(MvNormal(βΔ0[Mk]./d[Mk].+β0[Mk],Diagonal(τx*τy*τβ .*d[Mk].^2 .* a0[Mk]) |> inv), β[Mk])
      + sum(γ[Mk] .|> γj-> γj*log(ω)+(1.0-γj)*log(1-ω))
      + sum([:ylike, :xprior, :ϕprior, :ψprior,:νprior, :ωprior,:τyprior,:τxprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p))) 
    @assert lqγkMc1+c1 ≈ post "lqγkMc1+c1: $(lqγkMc1+c1); post: $post"

    lqγkMc2 = log(d[k]) - τx*τy*τβ*a0[k]/2*((β[k]-β0[k])^2*d[k]^2 - 2*(β[k]-β0[k])*βΔ0[k]d[k])+γ[k]*log(ω) + (1-γ[k])*log(1-ω)
    c2=c1 - τx*τy*τβ*a0[k]/2*βΔ0[k]^2
    @assert lqγkMc2+c2 ≈ post "lqγkMc2+c2: $(lqγkMc2+c2); post: $post"


    lp1=- τx*τy*τβ*a0[k]/2*((β[k]-β0[k])^2 - 2*(β[k]-β0[k])*βΔ0[k]) + log(ω)
    lp0=-log(v) - τx*τy*τβ*a0[k]/2*((β[k]-β0[k])^2/v^2 - 2*(β[k]-β0[k])*βΔ0[k]/v) + log(1-ω)
    lh = lp0 < lp1 ? lp1 : lp0

    lpγ[k] = (lp1-lh) - log(exp(lp1-lh)+exp(lp0-lh))
    lpγc[k] = (lp0-lh) - log(exp(lp1-lh)+exp(lp0-lh))
    
    pγ[k] = exp(lpγ[k])
    #@assert pγ[k] ≈ m.updateθpγk(;βk=β[k], β0k=β0[k],a0k=a0[k],τx,τy,v,ω)

    lqγkMc3 = γ[k]*lpγ[k] + (1-γ[k])*lpγc[k]
    c3=c2 + lh + log(exp(lp1-lh)+exp(lp0-lh))
    @assert lqγkMc3 + c3 ≈ post  "lqγkMc3 +c3: $(lqγkMc3+ c3); post: $post; k=$k, γ[k]= $(γ[k]) " 
  end

  @assert pγ ≈ m.updateθpγ(dgp).pγ
  @assert pγ ≈ (1:K .|> k-> m.updateθpγ(dgp,k,A0).pγ)
  @assert pγ ≈ (1:K .|> k-> m.updateθpγ(dgp,k,A0 |> Matrix).pγ)
  @assert lpγ ≈ m.updateθpγ(dgp; testupdateθpγ=true).pγtestinfo.lpγ

  #test the conditional draw (use repeated draws due to the discrete variable)
  for i ∈ 1:1000
    Random.seed!(i)
  
    manualdraw = m.draw(:γ; pγ)
    Random.seed!(i)
    conddraw = m.conditionaldraw(:γ_det; dgp).Θ.γ
    @assert all(manualdraw .== conddraw) "manualdraw=$manualdraw while conddraw=$conddraw"
  end
  
  #Now need to check the expectations
  #need placeholders for the updates
  D = m.formD(γ,hyper.v)
  μβ= #=inv(D) * =#β#inv(D)*β
  Λβ = (τx*τy .* D*hyper.A0*D)
  κ = κ0 + 0.1
  δ = δ0 + 0.2
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  αx=hyper.αx0+0.4
  ζx=hyper.ζx0+0.3
  αβ=hyper.αβ0-0.4
  ζβ=hyper.ζβ0-0.3

  

  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack μβ, Σβ= vbβ 
  vbτβ = m.VBqτβ(dgp; αβ, ζβ, testmoments=true)
  @unpack Eτβ= vbτβ    
  vbω = m.VBqω(dgp; κ, δ, testmoments=true)
  @unpack Elogω, Elog1Mω= vbω 


  dgpmom = m.DGP(dgp; β=vbβ, τx=vbτx, τy=vbτy, τβ=vbτβ, ω =vbω, strict=false)

  #compute and verify q
  @unpack pγ,pγtestinfo = m.updateθqγ(dgpmom; testupdateθqγ=true)
  @unpack lp̃γ1,lp̃γ0 = pγtestinfo

  #@info "timing diagonal q"
  #@btime m.updateθqγ($dgpmom)

  Eβ̃2 = diag(Σβ) .+ μβ.^2 .+ β0.^2 .- (2 .*β0.*μβ)
  lp̃γ1check=(-Eτx*Eτy*Eτβ .* a0) ./ 2 .* (Eβ̃2 .- 2(μβ-β0).*βΔ0) .+ Elogω
  lp̃γ0check=(-Eτx*Eτy*Eτβ .* a0) ./ 2 .* (Eβ̃2 ./ v^2 .- 2(μβ-β0).*βΔ0 ./ v) .+ Elog1Mω .- log(v)
  lhcheck = max.(lp̃γ0check, lp̃γ1check)
  lpγcheck = lp̃γ1check .- lhcheck .- log.(exp.(lp̃γ1check.- lhcheck) .+ exp.(lp̃γ0check.- lhcheck))
  lpγcheckc = lp̃γ1check .- lhcheck .- log.(exp.(lp̃γ1check.- lhcheck) .+ exp.(lp̃γ0check.- lhcheck))
  pγchecksafe = lpγcheck .|> exp
  lqγθsafe = (γ .* lpγcheck  .+ (1 .-γ) .* lpγcheckc) .|> exp

  #Old without the numerical approx
  #@eval Main γ, pγ,p̃γ1,p̃γ0 = $γ,$pγ,$p̃γ1,$p̃γ0
  p̃γ1check=exp.((-Eτx*Eτy*Eτβ .* a0) ./ 2 .* (Eβ̃2 .- 2(μβ-β0).*βΔ0) .+ Elogω)
  p̃γ0check=exp.((-Eτx*Eτy*Eτβ .* a0) ./ 2 .* (Eβ̃2 ./ v^2 .- 2(μβ-β0).*βΔ0 ./ v) .+ Elog1Mω .- log(v))
  pγcheck = p̃γ1check ./ (p̃γ1check .+ p̃γ0check)
  lqγθ = log.(pγ .^ γ .* (1 .- pγ).^(1 .-γ))
  qγθ = pγ .^ γ .* (1 .- pγ).^(1 .-γ)
  pγcheck  .= [isfinite(pγcheck[k]) ? pγcheck[k] : pγchecksafe[k] for k ∈ 1:K]

  @assert all((pγcheck .≈ pγ) .| ((pγcheck .+ 1.0) .≈ (pγ .+ 1.0))) "pγcheck=$pγcheck, pγ=$pγ"

  #only the expectation of the un-normalized value matters, so to check
  #the q distribution we need to create the probabilities from the expectations,
  #as opposed to taking the expetation of the probability.

  sim = Dict( :logω=>0.0,:log1Mω=>0.0, :lp̃γ0=>zeros(K), :lp̃γ1=>zeros(K))

  #d = (;αx,ζx, μβ,Λβ, μx, Λx,)
  #@eval Main d=$d
  #lqβθsims = Vector{Float64}(undef,iter)
  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τx=m.draw(:τx, ;αx,ζx),     
      τy=m.draw(:τy, ;αy,ζy), 
      β=m.draw(:β,; μβ,Λβ), 
      ω=m.draw(:ω; κ,δ),
      τβ=m.draw(:τβ; αβ, ζβ)
      
      )

    #simis = shuffle(1:K) .|> k->m.updateθpγ(dgpi,k,A0;testupdateθpγ=true) 

    #a bit overbuilt, could be refactored
    #the shuffled corresponds to the ordering of the function calls
    #unshuffler maps an unshuffled index to the appropriate call
    ks = shuffle(1:K)
    unshuffler = Dict(ks .=> 1:K)

    simis = (ks .|> k->m.updateθpγ(dgpi,k,A0;testupdateθpγ=true)) 
    @assert all((1:K .|> k -> ks[unshuffler[k]]) .== 1:K .== ks[sortperm(ks)])
    simis=simis[1:K .|> k -> unshuffler[k]]
      
    simi=(;pγ= simis .|> pγi->pγi.pγ,pγtestinfo = m.collapse(simis .|> pγi->pγi.pγtestinfo))


    sim[:logω] += log(dgpi.Θ.ω)
    sim[:log1Mω] += log(1-dgpi.Θ.ω)
    sim[:lp̃γ0] .+= simi.pγtestinfo.lp̃γ0
    sim[:lp̃γ1] .+= simi.pγtestinfo.lp̃γ1


  end

  for k ∈ keys(sim)
    sim[k] /= iter
  end

  lhsim = max.(sim[:lp̃γ1],sim[:lp̃γ0])

  pγsim = exp.(sim[:lp̃γ1]-lhsim) ./ (exp.(sim[:lp̃γ1]-lhsim) .+ exp.(sim[:lp̃γ0]-lhsim))
  qγsim = pγsim.^γ .* (1 .- pγsim).^(1 .- γ)
  lqγθsim = log.(qγsim)
  println("Elogω: sim $(sim[:logω]); act $(Elogω) ")  
  println("Elog1Mω: sim $(sim[:log1Mω]); act $(Elog1Mω) ")  
  #throw("clean up the abve garbage output")
  @info "*****lqγθ: $lqγθ; \nlqγθsim: $lqγθsim"
  @info "*****qγθ: $qγθ; \nqγθsim: $(qγsim)"

end

#allows for general A0
function testpostγ(dgp, A0::Matrix; iter)
  @info "testing γ (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack β0,βΔ0, v, κ0, δ0, αβ0, ζβ0 = hyper
  @unpack x, β, τy, τx, γ, ω, τβ = Θ
  @unpack F = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  D = m.formD(γ,v)
  Dinv = D |> inv
  d=D.diag

  #verify the prior and likelihood
  @assert lpdist(dgp,:βprior) ≈ (-(τy*τx*τβ/2)*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0)
    + 0.5*(K*log(τy*τx)-K*log(2π)+log(det(D*A0*D))+K*log(τβ)))
  @assert lpdist(dgp,:γprior) ≈ ((γ .|> γk-> log(ω^(γk)*(1-ω)^(1-γk))) |> sum)

  #verify the conditionals for each k
  pγ = missings(Float64, K)
  lpγ = similar(pγ)
  lpγc = similar(lpγ)
  for k ∈ 1:K
    Mk = Not(k)
    lqγkMc1 = log(d[k])-τx*τy*τβ/2*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0)+γ[k]*log(ω)+(1-γ[k])*log(1-ω)
    c1 = (K/2*log(τx*τy*τβ/2π) + sum(log.(d[Mk])) + 0.5*logdet(A0) + sum(γ[Mk] .|> γj-> γj*log(ω)+(1.0-γj)*log(1-ω))
      + sum([:ylike, :xprior, :ϕprior, :ψprior,:νprior, :ωprior,:τyprior,:τxprior, :τϕprior, :τβprior] .|> p->lpdist(dgp, p))) 



    lqγkMc2 = log(d[k]) - τx*τy*τβ/2*((β-β0)'D*A0*D*(β-β0)-2(β-β0)'*D*A0*βΔ0)+γ[k]*log(ω) + (1-γ[k])*log(1-ω)
    c2=c1 - τx*τy*τβ/2*βΔ0'A0*βΔ0
    @assert lqγkMc2+c2 ≈ post "lqγkMc2+c2: $(lqγkMc2+c2); post: $post"

    DGdk1 = deepcopy(D)
    DGdk1[k,k] = 1.0
    lp1= -τx*τy*τβ/2*((β-β0)'DGdk1*A0*DGdk1*(β-β0)-2(β-β0)'*DGdk1*A0*βΔ0) + log(ω)
    DGdkv = DGdk1 |> deepcopy
    DGdkv[k,k] = 1/v
    lp0= -log(v) - τx*τy*τβ/2*((β-β0)'DGdkv*A0*DGdkv*(β-β0)-2(β-β0)'*DGdkv*A0*βΔ0) + log(1-ω)
    
    #the following is designed to handle the underflow
    #the approach should be equiv to exp(lp1)/(exp(lp1)+exp(lp0))
    #basically, we want to avoid having to take a log of very small numbers
    lh = lp0 < lp1 ? lp1 : lp0
    lpγ[k] = lp1-lh - log(exp(lp0 - lh) + exp(lp1-lh))
    
    #another trick- note that log(1-p1/(p0+p1))=log(p0/(p0+1))
    #  =lp0-nfactor-log(exp(lp0 - nfactor) + exp(lp1-nfactor))
    lpγc[k] = lp0-lh - log(exp(lp0 - lh) + exp(lp1-lh))
    lqγkMc3 = γ[k] == 1 ? lpγ[k] : lpγc[k]

    pγ[k] = exp(lpγ[k])#exp(lp1-nfactor)/(exp(lp0 - nfactor) + exp(lp1-nfactor))
    #@info "lp1=$lp1, lp0=$lp0, lh=$lh, pγ[k]=$(pγ[k])" 
    #@assert pγ[k] ≈ m.updateθpγk(;βk=β[k], β0k=β0[k],a0k=a0[k],τx,τy,v,ω)


    c3=c2 + lh + log(exp(lp0 - lh) + exp(lp1-lh))
    @assert lqγkMc3 + c3 ≈ post  "lqγkMc3 +c3: $(lqγkMc3+ c3); c3=$c3, lqγkMc3=$lqγkMc3,
      post: $post; k=$k, γ[k]= $(γ[k]), pγ[k]=$(pγ[k]), lp0: $lp0, lp1: $lp1" 
  end


  θpγ = m.par_updateθpγ(dgp; testupdateθpγ=true)
  @eval Main t=$(θpγ)
  @assert pγ ≈ θpγ.pγ "pγ: $pγ; updateθpγ.pγ: $(θpγ.pγ)"
  @assert lpγ ≈ θpγ.pγtestinfo.lpγ "lpγ: $lpγ; updateθpγ.lpγ: $(θpγ.pγtestinfo.lpγ)"
  @assert lpγc ≈ θpγ.pγtestinfo.lpγc "lpγ: $lpγc; updateθpγ.lpγc: $(θpγ.pγtestinfo.lpγc)"
  @eval Main θpγ=$θpγ


  #test the conditional draw (use repeated draws due to the discrete variable)
  @info "testing coniditional draws for pγ=$pγ"
  for i ∈ 1:1_000
    Random.seed!(i)


    dgptest = deepcopy(dgp)
    manualdraw = dgptest.Θ.γ |> deepcopy
    for k ∈ 1:K
      manualdraw[k] = m.draw(:γ, m.updateθpγ(dgptest, k, A0).pγ)
      dgptest = m.DGP(dgptest, γ=manualdraw)
    end

    #println(manualdraw)
    Random.seed!(i)
    conddraw = m.conditionaldraw(:γ_det; dgp).Θ.γ
    @assert all(manualdraw .== conddraw) "manualdraw=$manualdraw while conddraw=$conddraw"
  end

  
  

  #Now need to check the expectations
  #need placeholders for the updates
  D = m.formD(γ,hyper.v)
  μβ= #=inv(D) * =#β#inv(D)*β
  Λβ = (τx*τy .* D*hyper.A0*D)
  κ = κ0 + 0.1
  δ = δ0 + 0.2
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  αx=hyper.αx0+0.4
  ζx=hyper.ζx0+0.3
  αβ=hyper.αβ0-0.4
  ζβ=hyper.ζβ0-0.3

  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack Σβ, Eβ̃= vbβ 
  vbγ = m.VBqγ(dgp; pγ, testmoments=true)
  @unpack ED, EDA0D= vbγ
  vbω = m.VBqω(dgp; κ, δ, testmoments=true)
  @unpack Elogω, Elog1Mω= vbω 
  vbτβ = m.VBqτβ(dgp; αβ, ζβ, testmoments=true)
  @unpack Eτβ= vbτβ    

  #compute and test the necessary moments
  ED = m.moment(:ED; pγ, v)  
  EDA0D = m.moment(:EDA0D; pγ, A0, v, ED)
  dgpmom = m.DGP(dgp; 
    β=vbβ, 
    τx=vbτx, 
    τy=vbτy,
    τβ=vbτβ, 
    γ=vbγ,
    ω=vbω, strict=false)

  for k ∈ 1:K
    m.updateθqγ(dgpmom, k, testupdateθqγ=true)
  end

  #compute and verify q
  qγ = m.par_updateθqγ(dgpmom; testupdateθqγ=true)
  qγθ = qγ.pγ .^ γ .* (1 .- qγ.pγ).^(1 .- γ)
  lqγθ=log.(qγθ)

  #@info "timing general q"
  #@btime [m.updateθqγ($dgpmom,k) for k ∈ 1:$K]

  qγmoments=Vector()
  for k ∈ 1:K
    #@info "testing $k/$K"
    EDGdk1k =  deepcopy(ED)
    EDGdkvk = deepcopy(ED)
    EDGdk1k[k,k] = 1.0
    EDGdkvk[k,k] = 1/v

    pγ1 = deepcopy(pγ)
    pγ0 = deepcopy(pγ)
    pγ1[k] = 1.0
    pγ0[k] = 0.0

    EDA0DGdk1k = m.moment(:EDA0D; pγ=pγ1, A0, v, ED=EDGdk1k)
    EDA0DGdkvk = m.moment(:EDA0D; pγ=pγ0, A0, v, ED=EDGdkvk)

    EdGdk1 =  EDGdk1k.diag
    EDA0DGdk1 = deepcopy(EDA0D)
    EDA0DGdk1[:,k] .=  EdGdk1 .* A0[:,k] .* EdGdk1[k]
    EDA0DGdk1[k,:] .=  EdGdk1 .* A0[k,:] .* EdGdk1[k]
    EDA0DGdk1[k,k] = EdGdk1[k]^2 * A0[k,k] 

    @eval Main EDA0DGdk1=$EDA0DGdk1
    @eval Main EDA0DGdk1k=$EDA0DGdk1k
    @assert all((EDA0DGdk1 .≈ EDA0DGdk1k) .| ((EDA0DGdk1 .+ 1.0) .≈ (EDA0DGdk1k .+ 1.0)))
    Eβ̃tDA0Dβ̃Gdk1kcheck = m.moment(Val{:Eβ̃tDA0Dβ̃}(); EDA0D=EDA0DGdk1, Eβ̃, Σβ)


    Eβ̃tDA0Dβ̃Gdk1k = tr(EDA0DGdk1k'*Σβ)+(μβ-β0)'*EDA0DGdk1k*(μβ-β0)
    Eβ̃tDA0Dβ̃Gdkvk = tr(EDA0DGdkvk'*Σβ)+(μβ-β0)'*EDA0DGdkvk*(μβ-β0)
    
    @assert Eβ̃tDA0Dβ̃Gdk1kcheck ≈ Eβ̃tDA0Dβ̃Gdk1k

    Eβ̃tDA0βΔ0Gdk1k = (μβ-β0)'*EDGdk1k*A0*βΔ0
    Eβ̃tDA0βΔ0Gdkvk = (μβ-β0)'*EDGdkvk*A0*βΔ0

    lp̃γ1k = -(Eτx*Eτy*Eτβ/2)*(Eβ̃tDA0Dβ̃Gdk1k-2*Eβ̃tDA0βΔ0Gdk1k)+Elogω
    lp̃γ0k = -(Eτx*Eτy*Eτβ/2)*(Eβ̃tDA0Dβ̃Gdkvk-2*Eβ̃tDA0βΔ0Gdkvk)+Elog1Mω-log(v)
    lh = max(lp̃γ1k,lp̃γ0k)

    lpγk = lp̃γ1k-lh-log(exp(lp̃γ1k-lh)+exp(lp̃γ0k-lh))
    lpγck = lp̃γ0k-lh-log(exp(lp̃γ1k-lh)+exp(lp̃γ0k-lh))
    pγk = exp(lpγk)
    pγck = exp(lpγck)

    @assert 1.0 ≈ pγk + pγck
    @assert (lp̃γ1k ≈ qγ.pγtestinfo.lp̃γ1[k]) || ((lp̃γ1k + 1.0) ≈ (qγ.pγtestinfo.lp̃γ1[k]+1.0))    
    @assert (lp̃γ0k ≈ qγ.pγtestinfo.lp̃γ0[k]) || (lp̃γ0k + 1.0 ≈ qγ.pγtestinfo.lp̃γ0[k]+1.0)
    @assert (lpγk ≈ qγ.pγtestinfo.lpγ[k]) || (lpγk + 1.0 ≈ qγ.pγtestinfo.lpγ[k]+1.0)
    @assert (lpγck ≈ qγ.pγtestinfo.lpγc[k]) || (lpγck + 1.0 ≈ qγ.pγtestinfo.lpγc[k]+1.0)
    @assert (pγk ≈ qγ.pγ[k]) || (pγk + 1.0 ≈ qγ.pγ[k]+1.0)
    @assert (pγck ≈ (1-qγ.pγ[k])) || (pγck + 1.0 ≈ (1.0 -qγ.pγ[k]) +1.0)

    push!(qγmoments,(;Eβ̃tDA0Dβ̃Gdk1k,Eβ̃tDA0Dβ̃Gdkvk ))


  end
  

  #only the expectation of the un-normalized value matters, so to check
  #the q distribution we need to create the probabilities from the expectations,
  #as opposed to taking the expetation of the probability.

  sim = Dict(:logω=>0.0,:log1Mω=>0.0, :β̃tDA0Dβ̃Gdk1=>zeros(K), :β̃tDA0Dβ̃Gdkv=>zeros(K), :lp̃γ0=>zeros(K), :lp̃γ1=>zeros(K))

  #d = (;αx,ζx, μβ,Λβ, μx, Λx,)
  #@eval Main d=$d
  #lqβθsims = Vector{Float64}(undef,iter)
  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τx=m.draw(:τx, ;αx,ζx),     
      τy=m.draw(:τy, ;αy,ζy), 
      β=m.draw(:β,; μβ,Λβ), 
      ω=m.draw(:ω; κ,δ),
      γ=m.draw(:γ;pγ),
      τβ=m.draw(:τβ; αβ, ζβ)
      
      )
    simi = m.par_updateθpγ(dgpi; testupdateθpγ=true) 



    sim[:logω] += log(dgpi.Θ.ω)
    sim[:log1Mω] += log(1-dgpi.Θ.ω)

    βsim = dgpi.Θ.β
    γsim = dgpi.Θ.γ

    for k ∈ 1:K
      γk1 = deepcopy(γsim )
      γk1[k] = 1.0
      γk0 = deepcopy(γsim )
      γk0[k] = 0.0

      D1 = m.formD(γk1,v)
      Dv = m.formD(γk0,v)

      sim[:β̃tDA0Dβ̃Gdk1][k] += (βsim-β0)'D1*A0*D1*(βsim-β0)
      sim[:β̃tDA0Dβ̃Gdkv][k] += (βsim-β0)'Dv*A0*Dv*(βsim-β0)
    end

    sim[:lp̃γ0] .+= simi.pγtestinfo.lp̃γ0
    sim[:lp̃γ1] .+= simi.pγtestinfo.lp̃γ1


  end

  for k ∈ keys(sim)
    sim[k] /= iter
  end
  lhsim = max.(sim[:lp̃γ1],sim[:lp̃γ0])

  pγsim = exp.(sim[:lp̃γ1]-lhsim) ./ (exp.(sim[:lp̃γ1]-lhsim) .+ exp.(sim[:lp̃γ0]-lhsim))
  qγθsim = pγsim.^γ .* (1 .- pγsim).^(1 .- γ)
  lqγθsim = log.(qγθsim)
  println("Elogω: sim $(sim[:logω]); act $(Elogω) ")  
  println("Elog1Mω: sim $(sim[:log1Mω]); act $(Elog1Mω) ")  
  println("Eβ̃tDA0Dβ̃Gdk1: sim $(sim[:β̃tDA0Dβ̃Gdk1]); \nact $([qγmoments[k].Eβ̃tDA0Dβ̃Gdk1k for k ∈ 1:K ]) ")  
  println("Eβ̃tDA0Dβ̃Gdkv: sim $(sim[:β̃tDA0Dβ̃Gdkv]); \nact $([qγmoments[k].Eβ̃tDA0Dβ̃Gdkvk for k ∈ 1:K ]) ")   
  println("Elog1Mω: sim $(sim[:log1Mω]); act $(Elog1Mω) ")  
  println("qγ.lp̃γ1: $(qγ.pγtestinfo.lp̃γ1), \nsim[:lp̃γ1]: $(sim[:lp̃γ1])")
  println("qγ.lp̃γ0: $(qγ.pγtestinfo.lp̃γ0), \nsim[:lp̃γ0]: $(sim[:lp̃γ0])")  
  @info "*****qγθ: $qγθ; \nlqγθsim: $qγθsim"
  @info "*****lqγθ: $lqγθ; \nlqγθsim: $lqγθsim"

end

testpostγ(dgp; kwargs...) = testpostγ(dgp, dgp.hyper.A0; kwargs...)

function testpostω(dgp; iter)
  @info "testing ω (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack A0, κ0, δ0 = hyper
  @unpack τy, τx, γ, ω = Θ
  @unpack F = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)

  #verify the prior and likelihood
  @assert lpdist(dgp,:γprior) ≈ sum(γ .* log(ω) .+ (1 .- γ).* log(1-ω))
  @assert lpdist(dgp,:ωprior) ≈ (κ0-1)*log(ω)+(δ0-1)*log(1-ω)-logbeta(κ0,δ0)

  lqω1Mc1 = (κ0-1)*log(ω) + (δ0-1)*log(1-ω) .+ sum(γ .* log(ω)) .+ sum((1 .- γ).*log(1-ω))
  c1=-logbeta(κ0,δ0) + sum([:ylike, :ϕprior, :xprior, :βprior, :ψprior,:νprior,:τyprior,:τxprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p)) 
  @assert lqω1Mc1+c1 ≈ post "lqω1Mc1+c1: $(lqω1Mc1+c1); post: $post"

  #conditional updating equations
  κ=κ0 .+ sum(γ)
  δ=δ0+K-sum(γ)
  @assert κ ≈ m.updateθpω(dgp).κ
  @assert δ ≈ m.updateθpω(dgp).δ

  #test the conditional draw
  Random.seed!(11)
  manualdraw = m.draw(:ω; κ, δ)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:ω; dgp).Θ.ω
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

  lqω2Mc1 = (κ-1) *log(ω) .+ (δ-1)*log(1-ω)
  @assert lqω2Mc1+c1 ≈ post "lqω2Mc1+c1: $(lqω2Mc1+c1); post: $post"

  lqω3Mc2 = logpdf(Beta(κ,δ),ω)
  c2 = c1 + logbeta(κ,δ)
  @assert post ≈ lqω3Mc2 + c2
  
  #Now need to check the expectations
  #need placeholders for the updates


  pγ = 1.0 .- exp.(-0.1 .* collect(1:K))
  vbγ = m.VBqγ(dgp; pγ, testmoments=true)
  #@unpack ED, EDA0D= vbγ

  dgpmom = m.DGP(dgp; γ=vbγ,strict=false)

  #compute and verify q(β)
  @unpack κ,δ = m.updateθqω(dgpmom)
  @assert κ ≈ κ0 + sum(pγ)
  @assert δ ≈ δ0 + K - sum(pγ)
  Eω = m.moment(:Eω; κ,δ)
  @assert Eω ≈ κ/(κ+δ)

  #note we subtract off the constant to allow for easy comparison
  lqωθ = logpdf(Beta(κ,δ),ω)+logbeta(κ,δ)
  @assert lqωθ ≈ (κ-1)*log(ω) + (δ-1)*log(1-ω)
  sim = Dict( :κ=>0.0,:δ=>0.0, :γ=>zeros(K))

  #d = (;αx,ζx, μβ,Λβ, μx, Λx,)
  #@eval Main d=$d
  lqωθsim = 0.0
  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      γ=m.draw(:γ; pγ))
    simi = m.updateθpω(dgpi)
    sim[:κ] += simi.κ
    sim[:δ] += simi.δ
    sim[:γ] .+= dgpi.Θ.γ
    lqωθsim += (simi.κ-1)*log(ω) + (simi.δ-1)*log(1-ω)
  end

  lqωθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end


  @info "*****lqωθ: $lqωθ; lqωθsim: $lqωθsim"

end

function testpostψ(dgp; iter, checkνnorm=true)
  @info "testing ψ (iter=$iter)"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack νmin, νmax, αν0, ζν0, lrν, rνdist = hyper
  @unpack x, β, τy, τx, ψ, ν = Θ
  @unpack F, r = data

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  Ψ = ψ |> m.formΨ

  #Φ= m.formΦ(ϕ; dims)
  #@assert (ỹ-XL*ϕ)'*(ỹ-XL*ϕ) ≈ (y-Φ*x)'*(y-Φ*x)


  #verify the prior and likelihood
  @assert lpdist(dgp,:ψprior) ≈ sum(ψ .|> (ψt)->(ν/2*log(ν/2) - log(gamma(ν/2))+(ν/2-1)*log(ψt)-ν*ψt/2))
  @assert lpdist(dgp,:xprior) ≈ (-(τy*τx/2)*((x-r)*sqrt(T)-F*β)'*Ψ*((x-r)*sqrt(T)-F*β)/T 
    + 0.5*(T*log(τy*τx) - T*log(2π)+log(det(Ψ))) - T/2*log(T))
  @assert lpdist(dgp,:νprior) ≈ αν0*log(ζν0) - loggamma(αν0) + (αν0-1)*log(ν) - ζν0*ν

  @unpack ζψ, αψ = m.updateθpψ(dgp)

  for t ∈ 1:T
    ft = F[t,:]
    lqψtMc1 = (-τy*τx*ψ[t]/2/T*((x[t]-r[t])*sqrt(T)- ft'*β)^2 + (ν/2-1/2)*log(ψ[t])-ν * ψ[t]/2)
    c1 = (0.5*log(τx*τy/(2π*T)) + ν/2*log(ν/2)-loggamma(ν/2)
      +sum(logpdf(Normal(F[j,:]'*β+r[j]*sqrt(T), 1/(τx*τy*ψ[j]/T)^0.5),x[j]*sqrt(T)) for j ∈ setdiff(1:T,t))
      +sum(logpdf(Gamma(ν/2, 1/(ν/2)),ψ[j]) for j ∈ setdiff(1:T,t))
      +sum([:ylike, :ϕprior, :γprior,:ωprior,:τyprior,:τxprior, :βprior, :νprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p)))
    
    @assert lqψtMc1 + c1 ≈ post
    @assert αψ[t] ≈ ν/2+1/2
    @assert ζψ[t] ≈  τx*τy/2*(x[t]^2 - 2x[t]*(ft'*β+r[t]*sqrt(T))/sqrt(T) + (β'*ft+r[t]*sqrt(T))^2/T)+ν/2

    lqψtMc2 = logpdf(Gamma(αψ[t], 1/ζψ[t]), ψ[t])
    c2 = c1 + loggamma(αψ[t])-αψ[t]*log(ζψ[t])
    @assert lqψtMc2 + c2 ≈ post "lqψtMc2+c2: $(lqψtMc2 + c2); post: $post" 
  end

  #test the conditional draw
  Random.seed!(11)
  manualdraw = m.draw(:ψ; αψ, ζψ)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:ψ; dgp).Θ.ψ
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

 
  #Now need to check the expectations
  #need placeholders for the updates
  μx=x.-0.1
  Λx=τy*τx .* m.formΨ(ψ)
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  αx=hyper.αx0+0.4
  ζx=hyper.ζx0+0.3
  ν = 4.0
  D = m.formD(Θ.γ,hyper.v)
  μβ= D*β
  Λβ = (τx*τy .* D*hyper.A0*D)*10
  @unpack lpν, pν, η1 = m.updateθpν(dgp)
  ntνnorm = updateθpν_normalized(dgp)

  vbx = m.VBqx(dgp; μx, Λx, testmoments=true)
  @unpack Ex2, Σx = vbx
  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy
  vbτx = m.VBqτx(dgp; αx, ζx, testmoments=true)
  @unpack Eτx= vbτx
  vbβ = m.VBqβ(dgp; μβ, Λβ, testmoments=true)
  @unpack EβtfPrT12ftβPrT12= vbβ 
  vbν = m.VBqν(dgp; lpν, pν, η1, testmoments=true)
  @unpack Eν= vbν
  

  #compute and test the necessary moments
  m.testmoment(:EβtfPrT12ftβPrT12; F, μβ, Λβ,r,dims)

  dgpmom = m.DGP(dgp; x=vbx, τy=vbτy, τx=vbτx, 
    β=vbβ, ν=vbν, strict=false)

  #test the approximation output
  @unpack αψ,ζψ = m.updateθqψ(dgpmom)
  @assert all(αψ .≈ Eν/2 .+ 0.5)
  for t ∈ 1:T
    f = F[t,:]
    @assert ζψ[t] ≈ (Eτx*Eτy/2*(Ex2[t] - 2μx[t]*(F[t,:]'*μβ+r[t]*sqrt(T))/sqrt(T) + EβtfPrT12ftβPrT12[t]/T)) .+ Eν/2
  end
  lqψθ = (αψ .- 1.0) .* log.(ψ) .- ζψ .* ψ
  lqψθcheck = (((αψt,ζψt, ψt)-> logpdf(Gamma(αψt,1/ζψt), ψt)).(αψ, ζψ, ψ) .- αψ .* log.(ζψ) .+ loggamma.(αψ))
  @assert lqψθ ≈ lqψθcheck

  lqψθsim = zeros(T)
  sim = Dict(:x2=>zeros(T), :βtfPrT12ftβPrT12=>zeros(T), :αψ=>zeros(T), :ζψ=>zeros(T), :ν=>0.0, :νnorm=>0.0)

  νtM1 = ν

  @eval Main νtM1 = $νtM1
  @eval Main lpν = $lpν
  @eval Main rνdist = $rνdist
  @eval Main lrν = $lrν

  


  #simualte the approximation output to make sure  wee are tkaing the expectations correctly
  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      τx=m.draw(:τx, ;αx,ζx), 
      τy=m.draw(:τy, ;αy,ζy), 
      β=m.draw(:β,; μβ,Λβ), 
      ν=m.draw(:ν; lpν, rνdist, lrν, νtM1), 
      x=m.draw(:x;μx, Λx),)
    
    νtM1 = dgpi.Θ.ν
    simi = m.updateθpψ(dgpi)
    lqψθsimi = (simi.αψ .- 1) .* log.(ψ) .- simi.ζψ .* ψ
    lqψθsim .+= lqψθsimi

    Θi = dgpi.Θ
    sim[:x2] .+= Θi.x .* Θi.x
    sim[:βtfPrT12ftβPrT12] .+=  ((f,rt)->(Θi.β'*f .+ rt*sqrt(T))*(f'*Θi.β .+ rt*(sqrt(T)))).(eachrow(F),r)
    sim[:ν] += Θi.ν

    checkνnorm && (sim[:νnorm] += m.draw(:ν; lpν=ntνnorm.lpν, rνdist, lrν, νtM1))
    sim[:αψ] .+= simi.αψ
    sim[:ζψ] .+= simi.ζψ



  end

  lqψθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  println("\nx2: sim $(sim[:x2]); \nact $(Ex2) ")
  println("\nβtfPrT12ftβPrT12: sim $(sim[:βtfPrT12ftβPrT12]); \nact $(EβtfPrT12ftβPrT12) ")
  println("\nαψ: sim $(sim[:αψ]); \nact $(αψ) ")
  println("\nζψ sim $(sim[:ζψ]); \nact $(ζψ) ")
  @info "\n*****lqψθ: $lqψθ; \nlqψθsim: $lqψθsim"
  println("\nν sim $(sim[:ν]); act $(Eν) ")
  checkνnorm && println("\nν sim $(sim[:νnorm]); ")

end

function updateθpν_normalized(dgp)
  @unpack Θ, dims,hyper= dgp
  @unpack T= dims
  @unpack αν0, ζν0, νmin, νmax = hyper
  @unpack ψ = Θ

  η1 = sum(log.(ψ) .- ψ) - 2ζν0
  lpν̃(ν) = (T*ν/2 + αν0-1)*log(ν/2) - T*loggamma(ν/2) + ν/2 * η1
  pν̃(ν) = lpν̃(ν) |> exp 

  η2 = m.moment(:η2;pν=pν̃, νmin, νmax ,)
  #@btime η2 = moment(Val{:η2}();pν=$pν̃, νmin=$νmin, νmax=$νmax ,)
  lpν(ν) = lpν̃(ν)*η2
  pν(ν) = lpν(ν) |> exp 

  return (; lpν, pν, η1)
end
function testpostν(dgp; iter)
  @info "testing ν (iter=$iter)"

  @warn "find and cite the paper with the η1/η2 integrals for ν"

  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack T,K = dims
  @unpack αν0, ζν0, νmin, νmax, rνdist, lrν = hyper
  @unpack τy, τx, ψ, ν = Θ

  #start by checking conditional on the moments
  post = lpdist(dgp,:post)
  Ψ = ψ |> m.formΨ


  #verify the prior and likelihood
  @assert lpdist(dgp,:ψprior) ≈ sum(ψ .|> (ψt)->(ν/2*log(ν/2) - log(gamma(ν/2))+(ν/2-1)*log(ψt)-ν*ψt/2))
  @info "truncated: = $((ν > νmax || ν<νmin) ? 0.0 : (αν0*log(ζν0)-loggamma(αν0)+(αν0-1)*log(ν) - ζν0*ν) - 
    log(cdf(Gamma(αν0,1/ζν0), νmax)-cdf(Gamma(αν0,1/ζν0), νmin)))"
  @info "truncateddist: = $(lpdist(dgp,:νprior))"

  ((ν>νmax) || (ν<νmin)) && throw("ν=$ν but ν ∈ $νmin:$νmax")
  denom = cdf(Gamma(αν0,1/ζν0), νmax)-cdf(Gamma(αν0,1/ζν0), νmin)

  @assert lpdist(dgp,:νprior) ≈ ((αν0*log(ζν0)-loggamma(αν0)+(αν0-1)*log(ν) - ζν0*ν)) - log(denom)

  #verify the conditionals
  lqνMc1 = T*ν/2*log(ν/2)-T*loggamma(ν/2) + sum((ν/2-1).*log.(ψ)-ν .* ψ ./ 2) + (αν0-1)*log(ν) - ζν0*ν
  c1 = αν0*log(ζν0)-loggamma(αν0) - log(denom) + 
    sum([:ylike, :ϕprior, :γprior,:ωprior,:τyprior,:τxprior,:βprior,:xprior, :τβprior, :τϕprior] .|> p->lpdist(dgp, p))
  @assert lqνMc1 + c1 ≈ post

  η1 = sum(log.(ψ)-ψ)-2ζν0
  lqνMc2 = (T*ν/2+αν0-1)*log(ν/2)-T*loggamma(ν/2)+ν/2*η1
  c2 = c1 - sum(log.(ψ)) + (αν0-1)*log(2)
  @assert lqνMc2 + c2 ≈ post
  @unpack lpν, pν = m.updateθpν(dgp)

  #=print("new update time")
  @btime m.updateθpν($dgp)

  print("old update time")
  @btime m.updateθpν_old($dgp)
  throw("stop")=#
  η1 = sum(log.(ψ) - ψ)-2ζν0
  lpνcheck = (T*ν/2+αν0-1)*log(ν/2) - T*loggamma(ν/2)+ν/2*η1

  @eval Main pν=$pν
  #@info("time to compute m.updateθpν(dgp):")
  #@btime m.updateθpν(dgp)
  #throw("stop")

  @assert lpν(ν) ≈ lpνcheck
  @assert pν(ν) ≈ lpνcheck |> exp
  η2 = m.moment(:η2; pν, νmin, νmax)
  c3 = c2 - log(η2)
  #Need to add η2 into the pdf, which was previously ommitted as a normalization constant
  @assert (lpν(ν) + log(η2) + c3) ≈ post
  
  #test the conditional draw
  Random.seed!(11)
  manualdraw = m.draw(:ν; νtM1=ν, lpν, rνdist, lrν)
  Random.seed!(11)
  conddraw = m.conditionaldraw(:ν; dgp).Θ.ν
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"

      
  #@info("time to compute η2:")
  #@btime m.moment(:η2; pν=$pν, νmin=$νmin, νmax=$νmax)



  #Now need to check the expectations
  #need placeholders for the updates
  αψ=[ν/2 for t ∈ 1:T]
  ζψ=[ν/2 for t ∈ 1:T]


  
  #compute and test the necessary moments
  vbψ = m.VBqψ(dgp;  αψ, ζψ, testmoments=true)
  @unpack Eψ, Elogψ= vbψ  

  dgpmom = m.DGP(dgp; ψ=vbψ, strict=false)

  @unpack lpν,pν = m.updateθqν(dgpmom)
  η1 = sum(Elogψ - Eψ) - 2ζν0
  pνcheck = (ν/2)^(T*ν/2+αν0-1)*gamma(ν/2)^(-T)*exp(ν*η1/2)
  @assert pν(ν) ≈ pνcheck
  @assert lpν(ν) ≈ log(pνcheck)
  lqνθ = lpν(ν)

  lqνθsim = 0.0
  sim = Dict(:logψ=>zeros(T), :ψ=>zeros(T),)
  
  for i ∈ 1:iter
    dgpi = m.DGP(dgp; 
      ψ=m.draw(:ψ, ;αψ,ζψ), 
      )
    
    simi = m.updateθpν(dgpi)
    lqνθsim += simi.lpν(ν)

    Θi = dgpi.Θ
    sim[:logψ] .+= log.(Θi.ψ)
    sim[:ψ] .+= Θi.ψ
  end

  lqνθsim /= iter
  for k ∈ keys(sim)
    sim[k] /= iter
  end

  println("\nlogψ : sim $(sim[:logψ]); \nact $(Elogψ) ")
  println("\nψ: sim $(sim[:ψ]); \nact $(Eψ) ")
  @info "\n*****lqνθ: $lqνθ; \nlqψθsim: $lqνθsim"

end


function testposty(dgp; iter)
  @info "testing y (iter=$iter)"
  #forget the moments for now- just check the conditional posterior. We will come back to the moments.
  @unpack data, Θ, dims,hyper = dgp
  @unpack S,P,T,R, s2t, Δt = dims
  @unpack ϕ0, M0 = hyper
  @unpack x, ϕ, τy, τx, ψ, β = Θ
  @unpack y, F, r = data

  #we will get the posterior density and check it against the derived conditional posterior density
  post = lpdist(dgp,:post)
  X̃L = m.formX̃L(;x, dims)
  X̃L_alt = m._formXL_alt(x;dims)*R
  Φ=m.formΦ(ϕ; dims)
  ỹ = m.formỹ(;y, x, dims)

  xS = sum(x[s2t.-l] for l ∈ 0:(Δt-1))

  @assert all((X̃L .≈ X̃L_alt) .| ((X̃L .+ 1.0) .≈ (X̃L_alt .+ 1.0)))

  #verify the predict functions
  @assert sum(Φ*x) ≈ m.derived(Val{:yx}(), dgp; testderived=true)[:sumyx]
  @assert sum(Φ*(F*β+r)) ≈ m.derived(Val{:yβ}(), dgp; testderived=true)[:sumyβ]

  #verify the prior and likelihood
  @assert lpdist(dgp,:ylike) ≈ (-(τy/2)*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ) +0.5*(S*log(τy)-S*log(2π)))

  #verify the algebra from the modelnotes documentation
  #step 1
  c1 = ((S/2*log(τy/(2π)))+lpdist(dgp, :ϕprior)+
    +lpdist(dgp, :xprior)+lpdist(dgp, :βprior)+lpdist(dgp, :γprior)+lpdist(dgp, :ωprior)
    +lpdist(dgp, :τyprior)+lpdist(dgp, :τxprior)+lpdist(dgp, :ψprior)+lpdist(dgp, :νprior)
    +lpdist(dgp, :τϕprior)+lpdist(dgp, :τβprior))
    lqy1Mc1 = -τy/2*(y-X̃L*ϕ-xS)'* (y-X̃L*ϕ-xS)
  @assert c1 + lqy1Mc1 ≈ post

  #now we can test the updating parameters
  Λy = I(S)*τy
  μy = X̃L*ϕ+xS

  @assert μy ≈ m.updatepy(dgp).μy
  @assert Λy ≈ m.updatepy(dgp).Λy
  @assert μy ≈ m.formΦ(ϕ;dims)*x

  #test the conditional draw
  #NOT SUPPORTED
  Random.seed!(11)
  manualdraw = m.draw(:y; μy, Σy=m.pdinv(Λy))
  Random.seed!(11)
  #=conddraw = m.conditionaldraw(:y; dgp).Θ.y
  @assert manualdraw ≈ conddraw "manualdraw=$manualdraw while conddraw=$conddraw"=#
  #quick replacement check
  @assert manualdraw ≈ m.draw(:y; μy=m.updatepy(dgp).μy, Σy=m.updatepy(dgp).Σy)



  lqy2Mc2 = logpdf(MultivariateNormal(μy,inv(Λy)),y)
  c2 = c1+S/2*log(2π)-0.5*log(det(Λy))#+0.5*(μy'*Λy*μy)
  @assert c2 + lqy2Mc2 ≈ post "c1 + lqy2Mc2=$(c1 + lqy2Mc2) but lqy2Mc2=$lqy2Mc2 !"


  ##Now need to check the expectations
  #need placeholders for the updates
  μx=x.-0.1
  Λx=τy*τx .* m.formΨ(ψ)
  αy=hyper.αy0+0.2
  ζy=hyper.ζy0+0.1
  μϕ=ϕ
  Λϕ=τy .* M0
  νtM1=  hyper.αν0/hyper.ζν0


  vbx = m.VBqx(dgp; μx, Λx, testmoments=true)
  @unpack EX̃L = vbx

  vbτy = m.VBqτy(dgp; αy, ζy, testmoments=true)
  @unpack Eτy= vbτy

  vbϕ = m.VBqϕ(dgp; μϕ, Λϕ, testmoments=true)
  @unpack μϕ,EΦ = vbϕ

  dgpmom = m.DGP(dgp; x=vbx, τy=vbτy, ϕ=vbϕ, strict=false)

  @unpack μy,Λy = m.updateqy(dgpmom,)
  ExS = sum(μx[(s2t .- (l - 1))] for l ∈ 1:Δt)
  @assert Λy ≈ I(S)*Eτy 
  @assert μy ≈ EX̃L*μϕ+ExS ≈ EΦ*μx
  lqy = -0.5*(y-μy)'*Λy*(y-μy)+0.5*(μy'*Λy*μy)

  lqysim = 0.0

  for i ∈ 1:iter
    dgpi = m.DGP(dgp; x=m.draw(:x;μx, Λx), τy=m.draw(:τy, ;αy,ζy))
    sim = m.updatepy(dgpi)
    lqysimi = -0.5*(y-sim.μy)'*sim.Λy*(y-sim.μy) + 0.5*sim.μy'*sim.Λy*sim.μy
    lqysim += lqysimi
  end

  lqysim /= iter

  #compare the results
  @info "*****lqy: $lqy; lqθsim: $lqysim"

end

function testupdates(;iter, kwargs...)
  m.getparameters!()
  dgp = formtestdata(; kwargs...)
  @eval Main dgp=$dgp


  #=testpostϕ(dgp; iter)
  testpostx(dgp; iter)
  testpostτy(dgp; iter)
  testpostτx(dgp; iter)
  testpostτϕ(dgp; iter)
  testpostτβ(dgp; iter)
  testpostβ(dgp; iter)
  testpostγ(dgp; iter)
  testpostω(dgp; iter)
  testpostψ(dgp; iter)
  testpostν(dgp; iter)=#
  testposty(dgp; iter)

  #the below is for testing the diagonal only variant of A0
  hyperdiagonalA0 = dgp.hyper(; strict=false, 
  β= (;β0=dgp.hyper.β0, βΔ0=dgp.hyper.βΔ0, v=dgp.hyper.v, A0=Diagonal(dgp.hyper.A0)))
  dgpdiagonalA0=m.DGP(; data=dgp.data, hyper=hyperdiagonalA0, dims=dgp.dims, Θ = dgp.Θ, records=dgp.records)
  @eval Main dgpdiagonalA0=$dgpdiagonalA0
  testpostγ(dgpdiagonalA0; iter)

  #m.testimhsampler(;iter)
  #m.testEν(dgp; iters=[1000,10^4, 10^5,], gausslimit = 100, 
  #  runprecisiontests=false, runtimingtests=true)

  #= checking which format is preferred
  @info "timing formΦ"
  Φ= zeros(dgp.dims.S,dgp.dims.T)
  @btime m.formΦ!($dgp.Θ.ϕ, $Φ; dims=$dgp.dims)

  @info "timing formXL"
  XL = zeros(dgp.dims.S,dgp.dims.P)
  @btime m.formXL!($dgp.Θ.x, $XL; dims=$dgp.dims) 
  =#

  return nothing
end

@time testupdates(; S=50, Δt=1, P=2, K=6, iter=10^4, )



  