#note currently this only applies to VB, but we could easily add more for more compelx non-vb parameters


############################ moments of q(ϕ) ##################

@kwdef struct VBqϕ{Tμϕ<:AbstractVector, TΛϕ<:AbstractMatrix, TΣϕ<:AbstractMatrix,
  TEΦ<:AbstractMatrix, TEΦtΦ<:AbstractMatrix, TEϕtM0ϕ<:Real} <: AbstractVBParameter
  
  μϕ::Tμϕ
  Λϕ::TΛϕ 
  Σϕ::TΣϕ

  EΦ::TEΦ
  EΦtΦ::TEΦtΦ
  EϕtM0ϕ::TEϕtM0ϕ
end

#This creates all moments that depend soley on qϕ
function VBqϕ(dgp::AbstractDGP; 
  μϕ,Λϕ, Σϕ=pdinv(Λϕ), testmoments=false)
  @unpack dims, hyper=dgp
  @unpack M0 = hyper

  if testmoments
    testmoment(:EΦtΦ; μϕ,Λϕ, dims)
    testmoment(:EϕtM0ϕ; μϕ, Λϕ, M0)
  end

  EΦ = moment(Val{:EΦ}(); dims, μϕ)
  EΦtΦ = moment(Val{:EΦtΦ}(); μϕ, Σϕ,  dims)
  EϕtM0ϕ = moment(Val{:EϕtM0ϕ}(); μϕ, Σϕ, M0)

  return VBqϕ(;μϕ, Λϕ, Σϕ, EΦ, EΦtΦ, EϕtM0ϕ)
end


############################ moments of q(x) #################

@kwdef struct VBqx{Tμx<:AbstractVector, TΛx<:AbstractMatrix, TΣx<:AbstractMatrix,
  TEXL<:AbstractMatrix, TEXLtXL<:AbstractMatrix, 
  TEX̃L<:AbstractMatrix, TEX̃LtX̃L <: AbstractMatrix,  
  TEX̃Ltỹ<:AbstractVector,TExS<:AbstractVector,
  TEỹ<:AbstractVector, TEỹtỹ<:Real, TEx2<:AbstractVector} <: AbstractVBParameter
  
  μx::Tμx
  Λx::TΛx 
  Σx::TΣx

  EXL::TEXL
  EXLtXL::TEXLtXL
  EX̃L::TEX̃L
  EX̃LtX̃L::TEX̃LtX̃L
  EX̃Ltỹ::TEX̃Ltỹ
  ExS::TExS
  Eỹ::TEỹ
  Eỹtỹ::TEỹtỹ
  Ex2::TEx2
end


#This creates all moments that depend soley on qx
function VBqx(dgp::AbstractDGP; 
  μx,Λx, Σx=pdinv(Λx), testmoments=false)
  @unpack dims, hyper, data=dgp
  @unpack y = data

  if testmoments
    testmoment(:Eỹtỹ; μx, Λx, y, dims)
    testmoment(:EX̃Ltỹ; μx,Λx,y, dims)
    testmoment(:EX̃LtX̃L; μx,Λx, dims)
  end

  EXL = moment(:EXL; μx, dims)
  EXLtXL = moment(:EXLtXL; μx, Σx, dims)
  EX̃L = moment(:EX̃L; EXL, dims,)
  EX̃LtX̃L = moment(:EX̃LtX̃L; EXLtXL, dims)
  EX̃Ltỹ = moment(:EX̃Ltỹ; dims, EXL, EXLtXL, y)
  ExS = moment(:ExS; μx,dims)
  Eỹ = moment(:Eỹ; y,μx,dims)
  Eỹtỹ = moment(:Eỹtỹ; EXLtXL, ExS, y, dims)
  Ex2 = moment(:Ex2; μx, Σx,)

  #@info EX̃Ltỹ |> typeof

  return VBqx(;μx,Λx, Σx, EXL, EXLtXL, EX̃L, EX̃LtX̃L, EX̃Ltỹ, ExS, Eỹ, Eỹtỹ, Ex2)
end

############################ moments of q(τy) #################

@kwdef struct VBqτy{Tαy<:Real, Tζy <:Real, TEτy<:Real} <: AbstractVBParameter
  
  αy::Tαy
  ζy::Tζy 

  Eτy::TEτy
end

#allow the kw testmoments argument for compatability
VBqτy(::AbstractDGP;  αy,ζy, testmoments=false) = VBqτy(;αy, ζy, Eτy=moment(:Eτy; αy, ζy))

############################ moments of q(τx) #################

@kwdef struct VBqτx{Tαx<:Real, Tζx <:Real, TEτx<:Real} <: AbstractVBParameter
  
  αx::Tαx
  ζx::Tζx 

  Eτx::TEτx
end

#allow the kw testmoments argument for compatability
VBqτx(::AbstractDGP;  αx,ζx, testmoments=false) = VBqτx(;αx, ζx, Eτx=moment(:Eτx; αx, ζx))


############################ moments of q(τϕ) #################
@kwdef struct VBqτϕ{Tαϕ <:Real, Tζϕ  <:Real, TEτϕ<:Real} <: AbstractVBParameter
  
  αϕ::Tαϕ
  ζϕ::Tζϕ 

  Eτϕ::TEτϕ
end

#allow the kw testmoments argument for compatability
VBqτϕ(::AbstractDGP;  αϕ,ζϕ, testmoments=false) = VBqτϕ(;αϕ, ζϕ, 
  Eτϕ=moment(:Eτϕ; αϕ, ζϕ))

############################ moments of q(τβ) #################
@kwdef struct VBqτβ{Tαβ<:Real, Tζβ <:Real, TEτβ<:Real} <: AbstractVBParameter
  
  αβ::Tαβ
  ζβ::Tζβ 

  Eτβ::TEτβ
end

#allow the kw testmoments argument for compatability
VBqτβ(::AbstractDGP;  αβ,ζβ, testmoments=false) = VBqτβ(;αβ, ζβ, 
  Eτβ=moment(:Eτβ; αβ, ζβ))
  

############################ moments of q(β) #################
@kwdef struct VBqβ{Tμβ<:AbstractVector, TΛβ<:AbstractMatrix, TΣβ<:AbstractMatrix,
  TEβ̃<:AbstractVector, TEβtfftβ, TEβtfPrT12ftβPrT12} <: AbstractVBParameter
  
  μβ::Tμβ
  Λβ::TΛβ 
  Σβ::TΣβ
  Eβ̃::TEβ̃

  EβtfPrftβPr::TEβtfftβ = nothing
  EβtfPrT12ftβPrT12::TEβtfPrT12ftβPrT12 = nothing
end

#This creates all moments that depend soley on qβ
function VBqβ(dgp::AbstractDGP; 
  μβ,Λβ, Σβ=pdinv(Λβ), testmoments=false)

  @unpack F,r = dgp.data
  @unpack β0 = dgp.hyper

  if testmoments
    testmoment(:EβtfPrftβPr; μβ, Λβ, F, r)
  end

  Eβ̃ = moment(:Eβ̃; μβ, β0)
  EβtfPrftβPr = moment(:EβtfPrftβPr; μβ, Σβ, F,r)

  return VBqβ(;μβ, Λβ, Σβ, Eβ̃, EβtfPrftβPr)
end

#SPECIAL VERSION for the GIRmodel
function VBqβ(dgp::AbstractDGP{<:DGPModelParametersGIR}; 
  μβ,Λβ, Σβ=pdinv(Λβ), testmoments=false)

  @unpack dims,data,hyper = dgp
  @unpack F,r = data
  @unpack β0 = hyper

  if testmoments
    testmoment(:EβtfPrT12ftβPrT12; μβ, Λβ, F, r, dims)
  end

  Eβ̃ = moment(:Eβ̃; μβ, β0)
  EβtfPrT12ftβPrT12 = moment(:EβtfPrT12ftβPrT12; μβ, Σβ, F,r, dims)

  return VBqβ(;μβ, Λβ, Σβ, Eβ̃, EβtfPrT12ftβPrT12)
end



############################ moments of q(γ) #################
@kwdef struct VBqγ{Tpγ<:AbstractVector, TED<:AbstractMatrix, TEDA0D<:AbstractMatrix, Tpγtestinfo} <: AbstractVBParameter
  
  pγ::Tpγ

  ED::TED
  EDA0D::TEDA0D

  pγtestinfo::Tpγtestinfo
end

#This creates all moments that depend soley on qϕ
function VBqγ(dgp::AbstractDGP; 
  pγ,testmoments=false, pγtestinfo=nothing)

  @unpack hyper, Θ = dgp
  @unpack A0, v = hyper

  if testmoments
    testmoment(:EDA0D; A0, pγ, v,)
  end

  ED = moment(:ED; pγ, v)
  EDA0D = moment(:EDA0D; A0, pγ, v, ED)

  return VBqγ(; pγ, ED, EDA0D, pγtestinfo)
end

############################ moments of q(ω) #################
@kwdef struct VBqω{Tκ<:Real, Tδ<:Real, TEω<:Real, TElogω<:Real, TElog1Mω<:Real} <: AbstractVBParameter
  
  κ::Tκ
  δ::Tδ

  Eω::TEω
  Elogω::TElogω
  Elog1Mω::TElog1Mω
end

function VBqω(dgp::AbstractDGP; κ, δ, testmoments=false,)

  Eω = moment(:Eω;  κ,δ)
  Elogω = moment(:Elogω;  κ,δ)
  Elog1Mω = moment(:Elog1Mω;  κ,δ)

  return VBqω(; κ, δ, Eω, Elogω, Elog1Mω)
end

############################ moments of q(ψ) #################
@kwdef struct VBqψ{Tαψ<:AbstractVector, Tζψ<:AbstractVector, 
  TEψ<:AbstractVector, TElogψ<:AbstractVector, TEΨ<:AbstractMatrix,} <: AbstractVBParameter
  
  αψ::Tαψ
  ζψ::Tζψ

  Eψ::TEψ
  Elogψ::TElogψ
  EΨ::TEΨ
end

function VBqψ(::AbstractDGP;  αψ, ζψ, testmoments=false,)

  Eψ = moment(:Eψ; αψ, ζψ)
  Elogψ = moment(:Elogψ; αψ, ζψ)
  EΨ = moment(:EΨ; Eψ)

  return VBqψ(; αψ, ζψ, Eψ, Elogψ, EΨ )
end

############################ moments of q(ν) #################
@kwdef struct VBqν{Tpν, Tlpν,
  Tη1<:Real, Tη2<:Real, TEν<:Real, } <: AbstractVBParameter
  
  pν::Tpν
  lpν::Tlpν

  η1::Tη1
  η2::Tη2
  Eν::TEν
end

function VBqν(dgp::AbstractDGP;  η1, pν, lpν, testmoments=false,)

  @unpack νmin, νmax, αν0, ζν0 = dgp.hyper

  if testmoments
    testmoment(:Eν; lpν, νmin, νmax, αν0, ζν0)
  end

  η2 = moment(:η2; pν, νmin, νmax)
  Eν = moment(:Eν;pν, νmax, νmin, η2)

  return VBqν(; pν, lpν, η1, η2, Eν)
end





#########moments of qy
#NOTE- if we are going to use this for missing values, we will need additional moments
@kwdef struct VBqy{Tμy<:AbstractVector, TΛy<:AbstractMatrix, TΣy<:AbstractMatrix,} <: AbstractVBParameter
  
  μy::Tμy
  Λy::TΛy 
  Σy::TΣy

end

VBqy(::AbstractDGP; 
  μy,Λy, Σy=pdinv(Λy), testmoments=false) = VBqy(; μy, Λy, Σy)


