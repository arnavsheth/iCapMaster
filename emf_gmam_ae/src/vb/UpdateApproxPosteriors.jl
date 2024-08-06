#this file contains varios functions related to updating

#############main updating equations
function updateθqϕ(dgp::AbstractDGP; testmoments=false)
  @unpack data, Θ, dims,hyper = dgp
  @unpack ϕ0, M0 = hyper
  @unpack EX̃L, EX̃LtX̃L,Eτy, EX̃Ltỹ = Θ

  Λϕ = Eτy *(EX̃LtX̃L+M0) |> Symmetric
  Σϕ = Λϕ |> pdinv
  μϕ = Eτy*Σϕ*(EX̃Ltỹ+M0*ϕ0)
  
  return VBqϕ(dgp;Λϕ, μϕ,Σϕ, testmoments)
end

function updateθqx(dgp::AbstractDGP; testmoments=false)
  @unpack data, Θ, dims,hyper = dgp
  @unpack Eτy, Eτx, EΨ, μβ, EΦ, EΦtΦ = Θ
  @unpack y,F,r = data 

  Λx = Eτy *(EΦtΦ+Eτx*EΨ) |> Symmetric
  Σx = Λx |> pdinv

  μx = Eτy*Σx*(EΦ'*y+Eτx*EΨ*(F*μβ+r))
  #@eval Main dgp, Λx, Σx, μx = $dgp, $Λx, $Σx, $μx 
  return VBqx(dgp;Λx, μx,Σx, testmoments)
end


function updateθqτy(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack S,T,K,P = dims
  @unpack EDA0D, EX̃LtX̃L,EϕtM0ϕ,EX̃L,Eτx,ED,Eỹtỹ,EX̃Ltỹ = Θ
  @unpack Σϕ, μϕ,μβ,Σβ, μx, Σx, EΨ, Eβ̃ = Θ
  @unpack αy0,ζy0,A0,M0,ϕ0,β0, βΔ0 =hyper
  @unpack F,r = data

  EϕtX̃LtX̃Lϕ = moment(:EϕtX̃LtX̃Lϕ; μϕ, Σϕ, EX̃LtX̃L,dims)
  ExtΨx = moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrtΨFβPr = moment(:EβtFtPrtΨFβPr; μβ, Σβ, F, EΨ,r)
  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)

  FμβPr=F*μβ+r
  αy=(S+T+K+P)/2+αy0
  ζy = (0.5*(Eỹtỹ+EϕtX̃LtX̃Lϕ-EX̃Ltỹ'*μϕ-μϕ'*EX̃Ltỹ) 
    + 0.5*(EϕtM0ϕ + ϕ0'*M0*ϕ0 - μϕ'*M0*ϕ0 - ϕ0'*M0*μϕ)
    + 0.5*Eτx*(ExtΨx + EβtFtPrtΨFβPr - μx'*EΨ*FμβPr-FμβPr'*EΨ*μx)
    + 0.5*Eτx*(Eβ̃tDA0Dβ̃+ βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζy0)

  return VBqτy(dgp;αy,ζy)
end

function updateθqτx(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack EDA0D,Eτy,ED = Θ
  @unpack μβ,Σβ,μx,Σx,EΨ, Eβ̃ = Θ
  @unpack αx0,ζx0,A0,β0, βΔ0 =hyper
  @unpack F,r = data


  ExtΨx = moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrtΨFβPr = moment(:EβtFtPrtΨFβPr; μβ, Σβ, F, EΨ,r)
  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)
  
  FμβPr=F*μβ+r     
  αx=(T+K)/2+αx0
  ζx = ( 
    + 0.5*Eτy*(ExtΨx + EβtFtPrtΨFβPr - μx'*EΨ*FμβPr-FμβPr'*EΨ*μx)
    + 0.5*Eτy*(Eβ̃tDA0Dβ̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζx0)

  return VBqτx(dgp;αx,ζx)
end


function updateθqβ(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack EDA0D,Eτy,Eτx,ED = Θ
  @unpack μx,EΨ = Θ
  @unpack A0,β0,βΔ0 =hyper
  @unpack F,r = data

  Λβ=Eτx*Eτy*(F'*EΨ*F + EDA0D)
  Σβ = Λβ |> pdinv
  μβ=Eτx*Eτy*Σβ*(F'EΨ*(μx-r)+EDA0D*β0+ED*A0*βΔ0)

  return VBqβ(dgp; μβ,Λβ,Σβ)
end

#diagonal only version
function updateθqγ(dgp::AbstractDGP, A0::Diagonal; testupdateθqγ=false,)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack μβ, Σβ, Eβ̃, Elogω, Elog1Mω, Eτx, Eτy = Θ
  @unpack A0,βΔ0,v =hyper
  @unpack F = data

  a0 = A0.diag
  
  Eβ̃2 = diag(Σβ) .+ Eβ̃.^2

  lp̃γ1 = (-Eτx*Eτy .*a0 ./2).*(Eβ̃2  .- 2Eβ̃ .* βΔ0) .+ Elogω
  lp̃γ0 = (-Eτx*Eτy .*a0 ./2).*(Eβ̃2 ./v^2 .- 2Eβ̃ .* βΔ0 ./ v) .+ Elog1Mω .- log(v)

  #the below normalization improves numerical properties
  lh = max.(lp̃γ1,lp̃γ0)
  pγ = exp.(lp̃γ1.-lh)./(exp.(lp̃γ1.-lh)+exp.(lp̃γ0.-lh))

  denom = log.(exp.(lp̃γ1.-lh)+exp.(lp̃γ0.-lh))
  lpγ = lp̃γ1.-lh .- denom
  lpγc = lp̃γ0.-lh .- denom
  

  #pγ = exp.(lp̃γ1) ./ (exp.(lp̃γ0) .+ exp.(lp̃γ1))
  pγtestinfo = testupdateθqγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)

  return (;pγ, pγtestinfo)
end

function updateθqγ(dgp::AbstractDGP, k, A0::Matrix; testupdateθqγ=false,)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack ED, EDA0D, μβ, Elogω, Elog1Mω, Eτx, Eτy, Σβ, Eβ̃ = Θ
  @unpack A0,β0,βΔ0,v =hyper
  @unpack F = data

  #start with ED|dk=1 and dk=1/v
  EDGdk1 = deepcopy(ED)
  EDGdk1[k,k] = 1.0
  EDGdkv = deepcopy(ED)
  EDGdkv[k,k] = 1/v


  #build-up to compute Eβ'DA0Dβ|dk=1
  EdGdk1 =  EDGdk1.diag
  EDA0DGdk1 = deepcopy(EDA0D)
  EDA0DGdk1[:,k] .=  EdGdk1 .* A0[:,k] .* EdGdk1[k]
  EDA0DGdk1[k,:] .=  EdGdk1 .* A0[k,:] .* EdGdk1[k]
  EDA0DGdk1[k,k] = EdGdk1[k]^2 * A0[k,k]  
  Eβ̃tDA0Dβ̃Gdk1 = moment(Val{:Eβ̃tDA0Dβ̃}(); EDA0D=EDA0DGdk1, Eβ̃, Σβ)

  #build-up to compute Eβ'DA0Dβ|dk=1/v (same procedure as dk=1)
  EdGdkv =  EDGdkv.diag
  EDA0DGdkv = deepcopy(EDA0D)
  EDA0DGdkv[:,k] .=  EdGdkv .* A0[:,k] .* EdGdkv[k]
  EDA0DGdkv[k,:] .=  EdGdkv .* A0[k,:] .* EdGdkv[k]
  EDA0DGdkv[k,k] = EdGdkv[k]^2 * A0[k,k]
  Eβ̃tDA0Dβ̃Gdkv = moment(Val{:Eβ̃tDA0Dβ̃}(); EDA0D=EDA0DGdkv, Eβ̃, Σβ)

  #build up to compute EβtDA0β0|dk=1 and dk=1/v
  Eβ̃tDA0βΔ0Gdk1 = Eβ̃'EDGdk1*A0*βΔ0
  Eβ̃tDA0βΔ0Gdkv = Eβ̃'EDGdkv*A0*βΔ0

  testupdateθqγ && testconditionalγkmoments(; 
    k,dgp, EDGdk1,EDGdkv,EDA0DGdk1,EDA0DGdkv,Eβ̃tDA0βΔ0Gdk1,Eβ̃tDA0βΔ0Gdkv,Eβ̃tDA0Dβ̃Gdk1,Eβ̃tDA0Dβ̃Gdkv)
  
  lp̃γ1 = -(Eτx*Eτy/2*(Eβ̃tDA0Dβ̃Gdk1-2*Eβ̃tDA0βΔ0Gdk1))+Elogω
  lp̃γ0 = -(Eτx*Eτy/2*(Eβ̃tDA0Dβ̃Gdkv-2*Eβ̃tDA0βΔ0Gdkv))+Elog1Mω-log(v)

  #the below normalization improves numerical properties
  lh = max(lp̃γ1,lp̃γ0)
  pγ = exp(lp̃γ1-lh)/(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))

  denom = log(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))
  lpγ = lp̃γ1-lh - denom
  lpγc = lp̃γ0-lh - denom

  pγtestinfo = testupdateθqγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)


  return (;pγ, pγtestinfo)
end



updateθqγ(dgp::AbstractDGP, args...; kwargs...) = updateθqγ(dgp, args..., dgp.hyper.A0; kwargs...)

# WARNING WARNING WARNING
# by running the below in parallel, we neglect the conditional dependence between γ
# this is NOT valid in the non-diagonal case
function par_updateθqγ(dgp::AbstractDGP; testupdateθqγ=false,)
  @unpack K = dgp.dims

  hand_pγs = 1:K .|> k->Threads.@spawn(updateθqγ(dgp,k; testupdateθqγ))
  pγs = hand_pγs .|> fetch

  pγ=pγs .|> pγi->pγi.pγ
  pγtestinfo = collapse(pγs .|> pγi->pγi.pγtestinfo)

  return VBqγ(dgp; pγ, pγtestinfo)
end


function updateθqω(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack pγ = Θ
  @unpack κ0,δ0 =hyper

  κ = κ0 + sum(pγ)
  δ = δ0+K-sum(pγ)

  return VBqω(dgp;κ,δ)
end

function updateθqψ(dgp::AbstractDGP)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack Eν, Eτx, Eτy, μx, Ex2, μβ, EβtfPrftβPr = Θ
  @unpack F,r = data

  αψ = fill(Eν/2 + 0.5,T)
  ζψ = Eτx*Eτy/2 .* (Ex2 - 2μx .* (F*μβ + r) + EβtfPrftβPr) .+ Eν/2

  return VBqψ(dgp; αψ, ζψ)
end

function updateθqν(dgp::AbstractDGP)
  @unpack data, Θ, dims, hyper = dgp
  @unpack T,K = dims
  @unpack Eψ, Elogψ = Θ
  @unpack αν0, ζν0 = hyper

  #reference Wand et al 2011
  η1 = sum(Elogψ-Eψ)-2ζν0
  lpν(ν) = (T*ν/2+αν0-1)*log(ν/2)-T*loggamma(ν/2)+ν/2*η1
  pν(ν) = lpν(ν) |> exp

  return VBqν(dgp;η1, pν, lpν)
end


function updateqy(dgp::AbstractDGP)
  @unpack data, Θ, dims, hyper = dgp
  @unpack EΦ, μx, Eτy = Θ
  @unpack αν0, ζν0 = hyper
  @unpack S=dims

  μy = EΦ*μx
  Λy = (I(S) .* Eτy)
  Σy = Λy\I

  return VBqy(dgp;μy, Λy, Σy,)
end
