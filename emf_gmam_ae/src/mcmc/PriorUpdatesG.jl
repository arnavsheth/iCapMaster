#these functions facilitate drawing from the joint prior distribution


function updateθpϕ(dgp::DGP{<:AbstractModelParametersG, <:AbstractData{<:NoData}})
  @unpack Θ, hyper = dgp
  @unpack ϕ0, M0 = hyper
  @unpack ϕ,τy, τϕ = Θ

  
  Λϕ =τy*τϕ .* M0 |> Symmetric
  Σϕ = Λϕ |> pdinv

  μϕ = ϕ0

  return (;Λϕ, μϕ, Σϕ)
end


function updateθpτy(dgp::DGP{<:AbstractModelParametersG, <:AbstractData{<:NoData}})
  @unpack Θ, dims,hyper= dgp
  @unpack P,K = dims
  @unpack ϕ,β,τy,τx,γ,τϕ,τβ = Θ
  @unpack αy0,ζy0,M0,β0,A0,ϕ0,v,βΔ0  =hyper


  D = formD(γ,v)
  Dinv = D |> inv

  αy=(P+K)/2+αy0
  ζy =(0.5*τϕ*(ϕ-ϕ0)'*M0*(ϕ-ϕ0)
    + 0.5*τx*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζy0)

  return (;αy,ζy)
end
  
function updateθpτx(dgp::DGP{<:AbstractModelParametersG, <:AbstractData{<:NoData}})
  @unpack Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack β,τy,τx,γ,τβ = Θ
  @unpack αx0,ζx0,β0,A0,v,βΔ0 =hyper

  D = formD(γ,v)
  Dinv = D |> inv

  αx=K/2+αx0
  ζx = 0.5*τy*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζx0

  return (;αx,ζx)
end

function updateθpβ(dgp::DGP{<:AbstractModelParametersG, <:AbstractData{<:NoData}})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τy,τx,γ,τβ = Θ
  @unpack β0,βΔ0,A0,v =hyper
  @unpack F,r = data

  D = formD(γ,v)

  Λβ=τx*τy*τβ .* D*A0*D
  Σβ=Λβ|>pdinv
  μβ=β0+pdinv(D)*βΔ0

  return (;μβ,Λβ,Σβ)
end

#we keep MH for code consistency though the proposal should generally be accepted
function updateθpν(dgp::DGP{<:AbstractModelParametersG, <:AbstractData{<:NoData}})
  @unpack Θ, dims,hyper= dgp
  @unpack αν0, ζν0 = hyper

  η1 = -2ζν0
  lpν(ν) = (αν0-1)*log(ν/2) + ν/2 * η1
  pν(ν) = lpν(ν) |> exp 

  return (; lpν, pν, η1)
end