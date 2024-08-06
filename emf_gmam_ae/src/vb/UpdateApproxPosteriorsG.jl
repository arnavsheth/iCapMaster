#this file contains varios functions related to updating

#############main updating equations
function updateθqϕ(dgp::DGP{<:AbstractModelParametersG}; testmoments=false)
  @unpack data, Θ, dims,hyper = dgp
  @unpack ϕ0, M0 = hyper
  @unpack EX̃L, EX̃LtX̃L,Eτy, EX̃Ltỹ, Eτϕ = Θ

  Λϕ = Eτy *(EX̃LtX̃L+M0*Eτϕ) |> Symmetric
  Σϕ = Λϕ |> pdinv
  μϕ = Eτy*Σϕ*(EX̃Ltỹ+M0*ϕ0*Eτϕ)
  
  return VBqϕ(dgp;Λϕ, μϕ,Σϕ, testmoments)
end



function updateθqτy(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack S,T,K,P = dims
  @unpack EDA0D, EX̃LtX̃L,EϕtM0ϕ,EX̃L,Eτx,ED,Eỹtỹ,EX̃Ltỹ = Θ
  @unpack Σϕ, μϕ,μβ,Σβ, μx, Σx, EΨ, Eβ̃, Eτϕ, Eτβ = Θ
  @unpack αy0,ζy0,A0,M0,ϕ0,β0,βΔ0   =hyper
  @unpack F,r = data

  EϕtX̃LtX̃Lϕ = moment(:EϕtX̃LtX̃Lϕ; μϕ, Σϕ, EX̃LtX̃L,dims)
  ExtΨx = moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrtΨFβPr = moment(:EβtFtPrtΨFβPr; μβ, Σβ, F, EΨ,r)
  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)

  FμβPr=F*μβ+r
  αy=(S+T+K+P)/2+αy0
  ζy = (0.5*(Eỹtỹ+EϕtX̃LtX̃Lϕ-EX̃Ltỹ'*μϕ-μϕ'*EX̃Ltỹ) 
    + 0.5*Eτϕ*(EϕtM0ϕ + ϕ0'*M0*ϕ0 - μϕ'*M0*ϕ0 - ϕ0'*M0*μϕ)
    + 0.5*Eτx*(ExtΨx + EβtFtPrtΨFβPr - μx'*EΨ*FμβPr-FμβPr'*EΨ*μx)
    + 0.5*Eτβ*Eτx*(Eβ̃tDA0Dβ̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζy0)

  return VBqτy(dgp;αy,ζy)
end


function updateθqτx(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack EDA0D,Eτy,ED = Θ
  @unpack μβ,Σβ,μx,Σx,EΨ, Eτβ,Eβ̃ = Θ
  @unpack αx0,ζx0,A0,β0,βΔ0 =hyper
  @unpack F,r = data


  ExtΨx = moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrtΨFβPr = moment(:EβtFtPrtΨFβPr; μβ, Σβ, F, EΨ,r)
  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)
  
  FμβPr=F*μβ+r     
  αx=(T+K)/2+αx0
  ζx = ( 
    + 0.5*Eτy*(ExtΨx + EβtFtPrtΨFβPr - μx'*EΨ*FμβPr-FμβPr'*EΨ*μx)
    + 0.5*Eτy*Eτβ*(Eβ̃tDA0Dβ̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζx0)

  return VBqτx(dgp;αx,ζx)
end



function updateθqτϕ(dgp::DGP{<:AbstractModelParametersG})
@unpack data, Θ, dims,hyper= dgp
@unpack P = dims
@unpack EϕtM0ϕ, Eτy, μϕ = Θ
@unpack αy0,ζy0,M0,ϕ0,αϕ0,ζϕ0 =hyper

    
αϕ=P/2+αϕ0
ζϕ =  0.5*Eτy*(EϕtM0ϕ + ϕ0'*M0*ϕ0 - μϕ'*M0*ϕ0 - ϕ0'*M0*μϕ) + ζϕ0

return VBqτϕ(dgp;αϕ,ζϕ)
end  


function updateθqτβ(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack K = dims
  @unpack EDA0D,Eτy,Eτx, ED = Θ
  @unpack μβ,Σβ,Eβ̃ = Θ
  @unpack αβ0,ζβ0,A0,β0, βΔ0 = hyper


  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)
      
  αβ=K/2+αβ0
  ζβ =  0.5*Eτy*Eτx*(Eβ̃tDA0Dβ̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζβ0

  return VBqτβ(dgp;αβ,ζβ)
end

function updateθqβ(dgp::DGP{<:AbstractModelParametersG})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack EDA0D,Eτy,Eτx,ED, Eτβ = Θ
  @unpack μx,EΨ = Θ
  @unpack A0,β0,βΔ0 =hyper
  @unpack F,r = data

  Λβ=Eτx*Eτy*(F'*EΨ*F + (Eτβ.*EDA0D))
  Σβ = Λβ |> pdinv
  μβ=Eτx*Eτy*Σβ*(F'EΨ*(μx-r)+(Eτβ.*EDA0D*β0)+(Eτβ.*ED*A0*βΔ0))

  return VBqβ(dgp; μβ,Λβ,Σβ)
end

#diagonal only version
function updateθqγ(dgp::DGP{<:AbstractModelParametersG}, A0::Diagonal; testupdateθqγ=false,)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack μβ, Σβ,Eβ̃, Elogω, Elog1Mω, Eτx, Eτy,Eτβ = Θ
  @unpack A0,β0,βΔ0, v =hyper
  @unpack F = data

  a0 = A0.diag
  Eβ̃2 = diag(Σβ) .+ Eβ̃.^2

  lp̃γ1 = (-Eτx*Eτy*Eτβ .*a0 ./2).*(Eβ̃2  .- 2Eβ̃.* βΔ0) .+ Elogω
  lp̃γ0 = (-Eτx*Eτy*Eτβ .*a0 ./2).*(Eβ̃2 ./v^2 .- 2Eβ̃ .* βΔ0 ./ v) .+ Elog1Mω .- log(v)

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

#TODO- a sequential update version?
function updateθqγ(dgp::DGP{<:AbstractModelParametersG}, k, A0::Matrix; testupdateθqγ=false,)
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack ED, EDA0D, μβ, Elogω, Elog1Mω, Eτx, Eτy, Σβ, Eτβ, Eβ̃ = Θ
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
  
    lp̃γ1 = -(Eτx*Eτy*Eτβ/2*(Eβ̃tDA0Dβ̃Gdk1-2*Eβ̃tDA0βΔ0Gdk1))+Elogω
    lp̃γ0 = -(Eτx*Eτy*Eτβ/2*(Eβ̃tDA0Dβ̃Gdkv-2*Eβ̃tDA0βΔ0Gdkv))+Elog1Mω-log(v)

  #the below normalization improves numerical properties
  lh = max(lp̃γ1,lp̃γ0)
  pγ = exp(lp̃γ1-lh)/(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))

  denom = log(exp(lp̃γ1-lh)+exp(lp̃γ0-lh))
  lpγ = lp̃γ1-lh - denom
  lpγc = lp̃γ0-lh - denom

  pγtestinfo = testupdateθqγ ? (;lpγ,lpγc,lp̃γ0,lp̃γ1) : (;)


  return (;pγ, pγtestinfo)
end

