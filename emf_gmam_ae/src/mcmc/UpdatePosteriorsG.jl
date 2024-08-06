
##############updating equations for the raw posterior, useful for simulation
#for now leave M0 fixed- but could make it dynamic by including X̃L'*X̃L
function updateθpϕ(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper = dgp
  @unpack ϕ0, M0 = hyper
  @unpack ϕ,x,τy, τϕ = Θ
  @unpack y = data 

  ỹ = formỹ(;y,x,dims)
  X̃L= formX̃L(;x, dims)
  Λϕ = τy*(X̃L'*X̃L+M0*τϕ) |> Symmetric
  Σϕ = Λϕ |> pdinv

  μϕ = τy*Σϕ*(X̃L'*ỹ+M0*ϕ0*τϕ)

  return (;Λϕ, μϕ, Σϕ)
end





function updateθpτy(dgp::DGP{<:AbstractModelParametersG})
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
    + 0.5*τx*(x-F*β-r)'*Ψ*(x-F*β-r) 
    + 0.5*τx*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζy0)

  return (;αy,ζy)
end


function updateθpτx(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ,τβ = Θ
  @unpack αx0,ζx0,β0,A0,v,βΔ0 =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv

  αx=(T+K)/2+αx0
  ζx = (0.5*τy*(x-F*β-r)'*Ψ*(x-F*β-r) 
    + 0.5*τy*τβ*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζx0) 

  return (;αx,ζx)
end

function updateθpτϕ(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack P = dims
  @unpack ϕ,τy = Θ
  @unpack αϕ0,ζϕ0, ϕ0,M0 =hyper


  αϕ=P/2+αϕ0
  ζϕ = (τy*0.5*(ϕ-ϕ0)'*M0*(ϕ-ϕ0) + ζϕ0)

  return (;αϕ,ζϕ)
end  

function updateθpτβ(dgp::DGP{<:AbstractModelParametersG})
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


function updateθpβ(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ,τβ = Θ
  @unpack β0,βΔ0,A0,v =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)

  Λβ=τx*τy*(F'*Ψ*F+(D*A0*D.*τβ))
  Σβ=Λβ|>pdinv
  μβ=τx*τy*pdinv(Λβ)*(F'*Ψ*(x-r)+ τβ.*(D*A0*(D*β0+βΔ0)))

  return (;μβ,Λβ,Σβ)
end


function updateθpγ(dgp::DGP{<:AbstractModelParametersG}, ::Diagonal; testupdateθpγ=false)
  @unpack Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack β,τy,τx,γ,ω,τβ = Θ
  @unpack β0,βΔ0,A0,v =hyper

  a0 = A0.diag
  β̃=formβ̃(β,β0)

  lp̃γ1 = -τx*τy/2*τβ .* a0 .* (β̃.^2 .- 2 .* β̃ .* βΔ0) .+ log(ω)
  lp̃γ0 =  -τx*τy/2*τβ .* a0 .* (β̃.^2 ./v^2 .- 2 .* β̃ .* βΔ0./v) .+ log((1-ω)/v) 

  #the below dramtically improves numerical properties by forcing the largest unnormalized value to 1
  lh = max.(lp̃γ1,lp̃γ0)
  denom=log.(exp.(lp̃γ1.-lh).+exp.(lp̃γ0.-lh))
  lpγ = lp̃γ1 .- lh .- denom
  lpγc = lp̃γ0 .- lh .- denom
  pγ = exp.(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end

function updateθpγ(dgp::DGP{<:AbstractModelParametersG}, k, ::Diagonal; testupdateθpγ=false)
  @unpack Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack β,τy,τx,γ,ω, τβ = Θ
  @unpack β0, βΔ0,A0,v =hyper

  β̃=formβ̃(β,β0)

  #θk = ((βk,β0k,a0k)->updateθpγk(;ω,τx,τy,v,βk,β0k,a0k)).(β,β0,a0)
  #@info "got here"

  lp̃γ1 = -τx*τy/2*τβ * A0[k,k] * (β̃[k]^2 - 2 * β̃[k] * βΔ0[k]) + log(ω)
  lp̃γ0 =  -τx*τy/2*τβ * A0[k,k] * (β̃[k]^2 /v^2 - 2 * β̃[k] * βΔ0[k]/v) + log((1-ω)/v) 

  #the below dramtically improves numerical properties by forcing the largest unnormalized value to 1
  lh = max(lp̃γ1,lp̃γ0)
  denom=log(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))
  lpγ = lp̃γ1 - lh - denom
  lpγc = lp̃γ0 - lh - denom
  pγ = exp(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end


function updateθpγ(dgp::DGP{<:AbstractModelParametersG},k, A0::Matrix; testupdateθpγ=false)
  @unpack Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack β,τy,τx,γ,ω, τβ = Θ
  @unpack β0,βΔ0,v =hyper

  #@info testupdateθpγ
  β̃=formβ̃(β,β0)

  #need to form the conditionals for D
  DGdk1= formDGγk(γ,v; k, γk=true)
  DGdk0= formDGγk(γ,v; k, γk=false)

  lp̃γ1 = -τx*τy*τβ/2 * (β̃'DGdk1*A0*DGdk1*β̃-2*β̃'DGdk1*A0*βΔ0) + log(ω)
  lp̃γ0 =  -τx*τy*τβ/2 * (β̃'DGdk0*A0*DGdk0*β̃-2*β̃'DGdk0*A0*βΔ0)  + log((1-ω)/v) 
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



