##############updating equations for the raw posterior, useful for simulation
function updateθpϕ(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper = dgp
  @unpack ϕ0, M0 = hyper
  @unpack ϕ,x,τy = Θ
  @unpack y = data 

  ỹ = formỹ(;y=data.y, x, dims)
  X̃L= formX̃L(;x, dims)
  Λϕ = τy*(X̃L'*X̃L+M0) |> Symmetric
  Σϕ = Λϕ |> pdinv

  μϕ = τy*Σϕ*(X̃L'*ỹ+M0*ϕ0)

  return (;Λϕ, μϕ, Σϕ)
end




function updateθpx(dgp::AbstractDGP; )
  @unpack data, Θ, dims= dgp
  @unpack x,ϕ, β, τy, τx, ψ = Θ
  @unpack F, y,r = data 

  Φ = formΦ(ϕ; dims)
  Ψ = ψ |> formΨ

  Λx=τy.*(Φ'*Φ+τx.*Ψ) |> Symmetric
  Σx = Λx |> pdinv
  μx=τy.*Σx*(Φ'*y+τx*Ψ*(F*β+r))

  return (;Λx, μx, Σx)
end


function updateθpτy(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack S,T,P,K = dims
  @unpack x,ϕ,β,τy,τx,ψ,γ = Θ
  @unpack αy0,ζy0,M0,β0,βΔ0,A0,ϕ0,v =hyper
  @unpack F,y,r = data

  X̃L = formX̃L(;x, dims)
  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv
  ỹ = formỹ(;y,x,dims)

  αy=(S+T+P+K)/2+αy0
  ζy = (0.5*(ỹ-X̃L*ϕ)'*(ỹ-X̃L*ϕ)
    + 0.5*(ϕ-ϕ0)'*M0*(ϕ-ϕ0)
    + 0.5*τx*(x-F*β-r)'*Ψ*(x-F*β-r) 
    + 0.5*τx*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζy0)

  return (;αy,ζy)
end

function updateθpτx(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ = Θ
  @unpack αx0,ζx0,β0,βΔ0, A0,v =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)
  Dinv = D |> inv

  αx=(T+K)/2+αx0
  ζx = (0.5*τy*(x-F*β-r)'*Ψ*(x-F*β-r) 
    + 0.5*τy*(β-β0-Dinv*βΔ0)'*D*A0*D*(β-β0-Dinv*βΔ0) + ζx0) 

  return (;αx,ζx)
end

function updateθpβ(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ψ,γ = Θ
  @unpack β0,βΔ0,A0,v =hyper
  @unpack F,r = data

  Ψ = ψ |> formΨ
  D = formD(γ,v)

  Λβ=τx*τy*(F'*Ψ*F+D*A0*D)
  Σβ=Λβ|>pdinv
  μβ=τx*τy*pdinv(Λβ)*(F'*Ψ*(x-r)+D*A0*(D*β0+βΔ0))

  return (;μβ,Λβ,Σβ)
end


function updateθpγ(dgp::AbstractDGP, ::Diagonal; testupdateθpγ=false)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τy,τx,γ,ω = Θ
  @unpack β0,βΔ0,A0,v =hyper

  a0 = A0.diag
  β̃ = formβ̃(β,β0)


  lp̃γ1 = -τx*τy/2 .* a0 .* (β̃.^2 .- 2 .* β̃ .* βΔ0) .+ log(ω)
  lp̃γ0 =  -τx*τy/2 .* a0 .* (β̃.^2 ./v^2 .- 2 .* β̃ .* βΔ0./v) .+ log((1-ω)/v) 

  #the below dramtically improves numerical properties by forcing the largest unnormalized value to 1
  lh = max.(lp̃γ1,lp̃γ0)
  denom=log.(exp.(lp̃γ1.-lh).+exp.(lp̃γ0.-lh))
  lpγ = lp̃γ1 .- lh .- denom
  lpγc = lp̃γ0 .- lh .- denom
  pγ = exp.(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end

function updateθpγ(dgp::AbstractDGP, k, ::Diagonal; testupdateθpγ=false)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τy,τx,γ,ω = Θ
  @unpack β0,βΔ0,A0,v =hyper

  β̃=formβ̃(β,β0)
  #θk = ((βk,β0k,a0k)->updateθpγk(;ω,τx,τy,v,βk,β0k,a0k)).(β,β0,a0)
  #@info "got here"


  lp̃γ1 = -τx*τy/2 * A0[k,k] * (β̃[k]^2 - 2 * β̃[k] * βΔ0[k]) + log(ω)
  lp̃γ0 =  -τx*τy/2 * A0[k,k] * (β̃[k]^2 /v^2 - 2 * β̃[k] * βΔ0[k]/v) + log((1-ω)/v) 

  #the below dramtically improves numerical properties by forcing the largest unnormalized value to 1
  lh = max(lp̃γ1,lp̃γ0)
  denom=log(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))
  lpγ = lp̃γ1 - lh - denom
  lpγc = lp̃γ0 - lh - denom
  pγ = exp(lpγ)

  pγtestinfo = testupdateθpγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (; pγ, pγtestinfo)
end


function updateθpγ(dgp::AbstractDGP,k, A0::Matrix; testupdateθpγ=false)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack β,τy,τx,γ,ω = Θ
  @unpack β0,βΔ0,v =hyper

  β̃=formβ̃(β,β0)

  #@info testupdateθpγ

  #need to form the conditionals for D
  DGdk1= formDGγk(γ,v; k, γk=true)
  DGdk0= formDGγk(γ,v; k, γk=false)

  lp̃γ1 = -τx*τy/2 * (β̃'DGdk1*A0*DGdk1*β̃-2*β̃'DGdk1*A0*βΔ0) + log(ω)
  lp̃γ0 =  -τx*τy/2 * (β̃'DGdk0*A0*DGdk0*β̃-2*β̃'DGdk0*A0*βΔ0)  + log((1-ω)/v) 
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

updateθpγ(dgp, args...; kwargs...) = updateθpγ(dgp, args..., dgp.hyper.A0; kwargs...)


# WARNING WARNING WARNING
# by running the below in parallel, we neglect the conditional dependence between γ
# this is NOT valid in the non-diagonal case
function par_updateθpγ(dgp::AbstractDGP; testupdateθpγ=false)
  @unpack K = dgp.dims

  hand_pγs = 1:K .|> k->Threads.@spawn(updateθpγ(dgp,k; testupdateθpγ))
  pγs = hand_pγs .|> fetch

  pγ=pγs .|> pγi->pγi.pγ
  pγtestinfo = collapse(pγs .|> pγi->pγi.pγtestinfo)

  return (; pγ, pγtestinfo)
end


function updateθpω(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack γ, ω = Θ
  @unpack κ0, δ0 =hyper

  κ=κ0+sum(γ)
  δ=δ0+K-sum(γ)
  return(;κ,δ)
end
  
function updateθpψ(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack x,β,τy,τx,ν = Θ
  @unpack F, r = data 

  αψ = fill(ν/2 + 0.5,T)
  ζψ = τx*τy/2 .* (x-F*β-r).^2 .+ ν/2

  return (; ζψ, αψ)
end


#unlike other update functions, this returns a function proportional to the pdf as opposed to pdf parameters

function updateθpν(dgp::AbstractDGP)
  @unpack Θ, dims,hyper= dgp
  @unpack T= dims
  @unpack αν0, ζν0 = hyper
  @unpack ψ = Θ

  η1 = sum(log.(ψ) .- ψ) - 2ζν0
  lpν(ν) = (T*ν/2 + αν0-1)*log(ν/2) - T*loggamma(ν/2) + ν/2 * η1
  pν(ν) = lpν(ν) |> exp 

  return (; lpν, pν, η1)
end


function updatepy(dgp::AbstractDGP, )

  @unpack Θ, dims, hyper, = dgp
  @unpack x, ϕ, τy = Θ
  @unpack S,T, s2t, Δt = dims 

  X̃L = formX̃L(; x, dims)
  xS = formxS(x; dims)


  μy = X̃L*ϕ + xS
  Λy = (I(S) .* τy)
  Σy = Λy\I

  return (; μy, Λy, Σy)
end



  