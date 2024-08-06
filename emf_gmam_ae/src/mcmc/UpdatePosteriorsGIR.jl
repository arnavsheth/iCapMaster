
##############updating equations for the raw posterior, useful for simulation
#for now leave M0 fixed- but could make it dynamic by including X̃L'*X̃L

function updateθpx(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims= dgp
  @unpack x,ϕ, β, τy, τx, ψ = Θ
  @unpack F, y,r = data 
  @unpack T = dims

  Φ = formΦ(ϕ; dims)
  Ψ = ψ |> formΨ

  Λx=τy.*(Φ'*Φ+τx.*Ψ) |> Symmetric
  Σx = Λx |> pdinv
  μx=τy.*Σx*(Φ'*y+τx/sqrt(T)*Ψ*(F*β+sqrt(T)*r))

  return (;Λx, μx, Σx)
end






function updateθpτy(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack S,T,P,K = dims
  @unpack x,ϕ,β,τy,τx,ψ,γ,τϕ,τβ = Θ
  @unpack αy0,ζy0,M0,β0,A0,ϕ0,v,βΔ0  =hyper
  @unpack F,y,r = data

  X̃L = formX̃L(;x, dims)
  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv
  ỹ = formỹ(;y,x,dims)

  αy=(S+T+P+K)/2+αy0
  ζy = (0.5*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ)
    + 0.5*τϕ*(ϕ-ϕ0)'*M0*(ϕ-ϕ0)
    + 0.5*τx/T*((x-r)*sqrt(T)-F*β)'*Ψ*((x-r)*sqrt(T)-F*β) 
    + 0.5*τx*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζy0)

  return (;αy,ζy)
end


function updateθpτx(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ,τβ = Θ
  @unpack αx0,ζx0,β0,A0,v,βΔ0 =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv

  αx=(T+K)/2+αx0
  ζx = (0.5*τy/T*(sqrt(T)*(x-r)-F*β)'*Ψ*(sqrt(T)*(x-r)-F*β)
    + 0.5*τy*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζx0) 

  return (;αx,ζx)
end

function updateθpτϕ(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack P = dims
  @unpack ϕ,τy = Θ
  @unpack αϕ0,ζϕ0, ϕ0,M0 =hyper


  αϕ=P/2+αϕ0
  ζϕ = (τy*0.5*(ϕ-ϕ0)'*M0*(ϕ-ϕ0) + ζϕ0)

  return (;αϕ,ζϕ)
end  

function updateθpτβ(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack β,τx,τy,γ = Θ
  @unpack β0, A0, αβ0,ζβ0,v, βΔ0 =hyper

  D = formD(γ,v)
  Dinv = D |> inv
  β̃=formβ̃(β,β0)

  #@eval Main (D,Dinv,A0,β,β0,ζgx0,τx,τy) = $(D,Dinv,A0,β,β0,ζgx0,τx,τy)
  αβ=K/2+αβ0
  ζβ = 0.5*τx*τy*(β̃-Dinv*βΔ0)'*D*A0*D*(β̃-Dinv*βΔ0) + ζβ0

  return (;αβ,ζβ)
end    


function updateθpβ(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ,τβ = Θ
  @unpack β0,βΔ0,A0,v =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)

  Λβ=τx*τy*(F'*Ψ*F/T+(D*A0*D.*τβ))
  Σβ=Λβ|>pdinv
  μβ=τx*τy*pdinv(Λβ)*(F'*Ψ*(x-r)/sqrt(T)+ τβ.*(D*A0*(D*β0+βΔ0)))

  return (;μβ,Λβ,Σβ)
end

  
function updateθpψ(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ν = Θ
  @unpack F, r = data 

  αψ = fill(ν/2 + 0.5,T)
  ζψ = τx*τy/2 .* ((x-r)*sqrt(T)-F*β).^2/T .+ ν/2

  return (; ζψ, αψ)
end



