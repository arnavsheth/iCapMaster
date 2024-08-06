#this file contains varios functions related to updating

#############main updating equations
function updateθqx(dgp::DGP{<:DGPModelParametersGIR}; testmoments=false)
  @unpack data, Θ, dims,hyper = dgp
  @unpack Eτy, Eτx, EΨ, μβ, EΦ, EΦtΦ = Θ
  @unpack y,F,r = data 
  @unpack T=dims

  Λx = Eτy *(EΦtΦ+Eτx*EΨ) |> Symmetric
  Σx = Λx |> pdinv

  μx = Eτy*Σx*(EΦ'*y+Eτx*EΨ*(F*μβ/sqrt(T)+r))
  #@eval Main dgp, Λx, Σx, μx = $dgp, $Λx, $Σx, $μx 
  return VBqx(dgp;Λx, μx,Σx, testmoments)
end


function updateθqτy(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack S,T,K,P = dims
  @unpack EDA0D, EX̃LtX̃L,EϕtM0ϕ,EX̃L,Eτx,ED,Eỹtỹ,EX̃Ltỹ = Θ
  @unpack Σϕ, μϕ,μβ,Σβ, μx, Σx, EΨ, Eβ̃, Eτϕ, Eτβ = Θ
  @unpack αy0,ζy0,A0,M0,ϕ0,β0,βΔ0   =hyper
  @unpack F,r = data

  EϕtX̃LtX̃Lϕ = moment(:EϕtX̃LtX̃Lϕ; μϕ, Σϕ, EX̃LtX̃L,dims)
  ExtΨx = moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrT12tΨFβPrT12 = moment(:EβtFtPrT12tΨFβPrT12; μβ, Σβ, F, EΨ,r,dims)
  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)

  FμβPr=F*μβ+r*sqrt(T)
  αy=(S+T+K+P)/2+αy0
  ζy = (0.5*(Eỹtỹ+EϕtX̃LtX̃Lϕ-EX̃Ltỹ'*μϕ-μϕ'*EX̃Ltỹ) 
    + 0.5*Eτϕ*(EϕtM0ϕ + ϕ0'*M0*ϕ0 - μϕ'*M0*ϕ0 - ϕ0'*M0*μϕ)
    + 0.5*Eτx*(ExtΨx + EβtFtPrT12tΨFβPrT12/T - μx'*EΨ*FμβPr/T-FμβPr'*EΨ*μx/T)
    + 0.5*Eτβ*Eτx*(Eβ̃tDA0Dβ̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζy0)

  return VBqτy(dgp;αy,ζy)
end


function updateθqτx(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack EDA0D,Eτy,ED = Θ
  @unpack μβ,Σβ,μx,Σx,EΨ, Eτβ,Eβ̃ = Θ
  @unpack αx0,ζx0,A0,β0,βΔ0 =hyper
  @unpack F,r = data


  ExtΨx = moment(:ExtΨx; μx, Σx, EΨ)
  EβtFtPrT12tΨFβPrT12 = moment(:EβtFtPrT12tΨFβPrT12; μβ, Σβ, F, EΨ,r,dims)
  Eβ̃tDA0Dβ̃ = moment(:Eβ̃tDA0Dβ̃; Eβ̃, Σβ, EDA0D,)
  
  FμβPr=F*μβ+r*sqrt(T)     
  αx=(T+K)/2+αx0
  ζx = ( 
    + 0.5*Eτy*(ExtΨx + EβtFtPrT12tΨFβPrT12/T - μx'*EΨ*FμβPr/sqrt(T)-FμβPr'*EΨ*μx/sqrt(T))
    + 0.5*Eτy*Eτβ*(Eβ̃tDA0Dβ̃ + βΔ0'*A0*βΔ0 - βΔ0'*A0*ED*Eβ̃ - Eβ̃'*ED*A0*βΔ0) + ζx0)

  return VBqτx(dgp;αx,ζx)
end



function updateθqτϕ(dgp::DGP{<:DGPModelParametersGIR})
@unpack data, Θ, dims,hyper= dgp
@unpack P = dims
@unpack EϕtM0ϕ, Eτy, μϕ = Θ
@unpack αy0,ζy0,M0,ϕ0,αϕ0,ζϕ0 =hyper

    
αϕ=P/2+αϕ0
ζϕ =  0.5*Eτy*(EϕtM0ϕ + ϕ0'*M0*ϕ0 - μϕ'*M0*ϕ0 - ϕ0'*M0*μϕ) + ζϕ0

return VBqτϕ(dgp;αϕ,ζϕ)
end  


function updateθqτβ(dgp::DGP{<:DGPModelParametersGIR})
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

function updateθqβ(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack EDA0D,Eτy,Eτx,ED, Eτβ = Θ
  @unpack μx,EΨ = Θ
  @unpack A0,β0,βΔ0 =hyper
  @unpack F,r = data

  Λβ=Eτx*Eτy*(F'*EΨ*F/T + (Eτβ.*EDA0D))
  Σβ = Λβ |> pdinv
  μβ=Eτx*Eτy*Σβ*(F'EΨ*(μx-r)/sqrt(T)+(Eτβ.*EDA0D*β0)+(Eτβ.*ED*A0*βΔ0))

  return VBqβ(dgp; μβ,Λβ,Σβ)
end

function updateθqψ(dgp::DGP{<:DGPModelParametersGIR})
  @unpack data, Θ, dims,hyper= dgp
  @unpack T,K = dims
  @unpack Eν, Eτx, Eτy, μx, Ex2, μβ, EβtfPrT12ftβPrT12 = Θ
  @unpack F,r = data

  αψ = fill(Eν/2 + 0.5,T)
  ζψ = Eτx*Eτy/2 .* (Ex2 - 2μx .* (F*μβ + r*sqrt(T))/sqrt(T) + EβtfPrT12ftβPrT12/T) .+ Eν/2

  return VBqψ(dgp; αψ, ζψ)
end
