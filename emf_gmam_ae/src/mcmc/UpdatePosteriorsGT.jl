
##############updating equations for the raw posterior, useful for simulation
#for now leave M0 fixed- but could make it dynamic by including X̃L'*X̃L
function updateθpτy(dgp::DGP{<:DGPModelParametersGT})
  @unpack data, Θ, dims,hyper= dgp
  @unpack S,T,P,K = dims
  @unpack x,ϕ,β,τy,ψ,γ,τϕ,τβ = Θ
  @unpack αy0,ζy0,M0,β0,A0,ϕ0,v,βΔ0  =hyper
  @unpack F,y,r = data

  X̃L = formX̃L(;x, dims)
  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv
  ỹ = formỹ(;y,x,dims)

  αy=(S+P)/2+αy0
  ζy = (0.5*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ)
    + 0.5*τϕ*(ϕ-ϕ0)'*M0*(ϕ-ϕ0) + ζy0)

  return (;αy,ζy)
end


function updateθpx(dgp::DGP{<:DGPModelParametersGT}; )
  @unpack data, Θ, dims= dgp
  @unpack x,ϕ, β, τy, τx, ψ = Θ
  @unpack F, y,r = data 

  Φ = formΦ(ϕ; dims)
  Ψ = ψ |> formΨ

  Λx=(τy.*Φ'*Φ+τx.*Ψ) |> Symmetric
  Σx = Λx |> pdinv
  μx=Σx*(τy.*Φ'*y+τx*Ψ*(F*β+r))

  return (;Λx, μx, Σx)
end



function updateθpτx(dgp::DGP{<:DGPModelParametersGT})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τx,ψ,γ,τβ = Θ
  @unpack αx0,ζx0,β0,A0,v,βΔ0 =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv

  αx=(T+K)/2+αx0
  ζx = (0.5*(x-F*β-r)'*Ψ*(x-F*β-r) 
    + 0.5*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζx0) 

  return (;αx,ζx)
end

function updateθpτβ(dgp::DGP{<:DGPModelParametersGT})
  @unpack data, Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack β,τx,γ = Θ
  @unpack β0, A0, αβ0,ζβ0,v, βΔ0 =hyper

  D = formD(γ,v)
  Dinv = D |> inv
  β̃=formβ̃(β,β0)

  #@eval Main (D,Dinv,A0,β,β0,ζgx0,τx,τy) = $(D,Dinv,A0,β,β0,ζgx0,τx,τy)
  αβ=K/2+αβ0
  ζβ = 0.5*τx*(β̃-Dinv*βΔ0)'*D*A0*D*(β̃-Dinv*βΔ0) + ζβ0

  return (;αβ,ζβ)
end    


function updateθpβ(dgp::DGP{<:DGPModelParametersGT})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ,τβ = Θ
  @unpack β0,βΔ0,A0,v =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)

  Λβ=τx*(F'*Ψ*F+(D*A0*D.*τβ))
  Σβ=Λβ|>pdinv
  μβ=τx*pdinv(Λβ)*(F'*Ψ*(x-r)+ τβ.*(D*A0*(D*β0+βΔ0)))

  return (;μβ,Λβ,Σβ)
end


function updateθpγ(dgp::DGP{<:DGPModelParametersGT}, ::Diagonal; testupdateθpγ=false)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τx,γ,ω,τβ = Θ
  @unpack β0,βΔ0,A0,v =hyper

  a0 = A0.diag
  β̃=formβ̃(β,β0)

  lp̃γ1 = -τx/2*τβ .* a0 .* (β̃.^2 .- 2 .* β̃ .* βΔ0) .+ log(ω)
  lp̃γ0 =  -τx/2*τβ .* a0 .* (β̃.^2 ./v^2 .- 2 .* β̃ .* βΔ0./v) .+ log((1-ω)/v) 

  #the below dramtically improves numerical properties by forcing the largest unnormalized value to 1
  lh = max.(lp̃γ1,lp̃γ0)
  denom=log.(exp.(lp̃γ1.-lh).+exp.(lp̃γ0.-lh))
  lpγ = lp̃γ1 .- lh .- denom
  lpγc = lp̃γ0 .- lh .- denom
  pγ = exp.(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end

function updateθpγ(dgp::DGP{<:DGPModelParametersGT}, k, ::Diagonal; testupdateθpγ=false)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τx,γ,ω, τβ = Θ
  @unpack β0, βΔ0,A0,v =hyper

  β̃=formβ̃(β,β0)

  #θk = ((βk,β0k,a0k)->updateθpγk(;ω,τx,τy,v,βk,β0k,a0k)).(β,β0,a0)
  #@info "got here"

  lp̃γ1 = -τx/2*τβ * A0[k,k] * (β̃[k]^2 - 2 * β̃[k] * βΔ0[k]) + log(ω)
  lp̃γ0 =  -τx/2*τβ * A0[k,k] * (β̃[k]^2 /v^2 - 2 * β̃[k] * βΔ0[k]/v) + log((1-ω)/v) 

  #the below dramtically improves numerical properties by forcing the largest unnormalized value to 1
  lh = max(lp̃γ1,lp̃γ0)
  denom=log(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))
  lpγ = lp̃γ1 - lh - denom
  lpγc = lp̃γ0 - lh - denom
  pγ = exp(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end


function updateθpγ(dgp::DGP{<:DGPModelParametersGT},k, A0::Matrix; testupdateθpγ=false)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τx,γ,ω, τβ = Θ
  @unpack β0,βΔ0,v =hyper

  #@info testupdateθpγ
  β̃=formβ̃(β,β0)

  #need to form the conditionals for D
  DGdk1= formDGγk(γ,v; k, γk=true)
  DGdk0= formDGγk(γ,v; k, γk=false)

  lp̃γ1 = -τx*τβ/2 * (β̃'DGdk1*A0*DGdk1*β̃-2*β̃'DGdk1*A0*βΔ0) + log(ω)
  lp̃γ0 =  -τx*τβ/2 * (β̃'DGdk0*A0*DGdk0*β̃-2*β̃'DGdk0*A0*βΔ0)  + log((1-ω)/v) 
  lh = max(lp̃γ1,lp̃γ0)
  #the log part should evaluate to 0 in the overflow/underflow scenario
  # equiv to pγ = exp.(lp̃γ1) ./ (exp.(lp̃γ0) .+ exp.(lp̃γ1))
  denom=log(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))
  lpγ = lp̃γ1 - lh - denom
  lpγc = lp̃γ0 - lh - denom
  pγ = exp(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end

  
function updateθpψ(dgp::DGP{<:DGPModelParametersGT})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τx,ν = Θ
  @unpack F, r = data 

  αψ = fill(ν/2 + 0.5,T)
  ζψ = τx/2 .* (x-F*β-r).^2 .+ ν/2

  return (; ζψ, αψ)
end



