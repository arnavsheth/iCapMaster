#I expect that this file will be broken into smaller files at some point


#generates the S×T matrix
function formΦ!(ϕ::AbstractVector, Φ::Matrix; dims)
    @unpack S,T,P, s2allts, s2t, Δt, ιϕ = dims
    @assert (S,T) == size(Φ)
    Φ .= 0.0
    ϕ̃ = formϕ̃(; ϕ, dims)

    for s ∈ 1:S
      ts = s2allts[s]
      Φ[s, ts] .= ϕ̃
    end
  
    return Φ
  end
  
  formϕ̃(; ϕ, dims) = [ϕ; ones(dims.Δt) - dims.ιϕ'ϕ]
  formΦ(ϕ::AbstractVector; dims) = formΦ!(ϕ, zeros(dims.S, dims.T); dims)

  function testΦ(ϕ::AbstractVector; dims)
    @unpack S,T,P, Δt, s2t, ιϕ = dims
    Φ = formΦ(ϕ; dims)

    for s ∈ 1:S, t ∈ 1:T
      if 1 ≤ P-(s2t[s]-Δt-t) ≤ P
        @assert Φ[s,t] ≈ ϕ[P-(s2t[s]-Δt-t)]
      elseif s2t[s]-Δt< t ≤ s2t[s]
        @assert Φ[s,t] ≈ 1-ιϕ[:, Δt - (s2t[s]-t)]'ϕ
      else
        @assert Φ[s,t] == 0.0
      end 
    end

    return nothing

  end
  
  #generate the S×P matrix XL
  function _formXL!(x::AbstractVector{Tx}, XL::AbstractMatrix{Tx}; dims) where Tx
    
    
    #sanity check on dimensionality
  
    @unpack T,S,P, s2allts, Δt = dims
    @assert T == length(x) "T: $T, length(x): $(length(x)) x: $x"
    @assert (S,P+Δt) == size(XL)
  
    for s ∈ 1:S
      ts = s2allts[s]
      XL[s,:] .= @view (x[ts])
    end
  
    return XL
  end
  
  _formXL(x::AbstractVector; dims,) = _formXL!(x, zeros(dims.S, dims.P+dims.Δt); dims)

formX̃L(; x=nothing, dims,XL=_formXL(x::AbstractVector; dims,)) = formX̃L!(zeros(dims.S, dims.P); XL, dims)
function formX̃L!(X̃L::AbstractMatrix; dims, XL) 
    X̃L .= XL*dims.R
  return X̃L
end


function formX̃L1step(;x,Ex,dims)
  @unpack Δt, R, P = dims

  XLEx = _formXL(Ex; dims)
  XLx = _formXL(x; dims)

  X̃L1step = [XLx[:,1:P] XLEx[:,(P+1):(P+Δt)]]*R
  
  return X̃L1step
end



function formxS(x; dims)
  @unpack s2t, Δt = dims
  return sum(x[s2t .- l] for l ∈ 0:(Δt-1))
end

function formỹ(;y,x=nothing,dims, xS = formxS(x; dims), ỹ=zeros(dims.S))
  @unpack S,T,s2t, Δt = dims
  @assert  S == length(y) == length(ỹ) == length(xS)

  ỹ .= y .- xS

  return ỹ
end

formβ̃(β,β0) = β - β0


formΨ(ψ::AbstractVector) = Diagonal(ψ)

formd(zk::Bool, v::Float64) = (zk + (1-zk)/v^2)^0.5
formD(d::AbstractVector{Float64}) = Diagonal(d)
formD(γ::Vector{Bool}, v::Float64) = formd.(γ,v) |> formD
formDGγk(γ::Vector{Bool}, v::Float64; k, γk::Bool) = [
    ifelse(j==k, γk, γ[j]) |> γj-> formd(γj,v) for j ∈ 1:length(γ)] |> formD

logγpdf(γ; lpγ::Float64, lpγc::Float64) = γ * lpγ + (1-γ)*lpγc


  ###################methods only for testing purposeds################

#only for testing purposes
#generate the S×P matrix XL (using the sparse matrices)
function _formXL_alt!(x::AbstractVector{Tx}, XL::AbstractMatrix{Tx}; dims) where Tx
  
  
    #sanity check on dimensionality
  
    @unpack T,S,P, s2t, ιXL, Δt = dims
    @assert T == length(x) "T: $T, length(x): $(length(x)) x: $x"
    @assert (S,P+ Δt) ≡ size(XL)
    @assert length(ιXL) ≡ P+ Δt
  
    XL = reduce(hcat, (ιXL .|> ιXLp->ιXLp*x))

  
    return XL
  end
  
  _formXL_alt(x::AbstractVector; dims,) = _formXL_alt!(x, zeros(dims.S, dims.P+dims.Δt); dims)
