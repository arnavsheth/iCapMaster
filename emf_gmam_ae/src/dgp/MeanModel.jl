#define the relevant types

#dimensions 
struct MeanModelDims{TFθ2name,Ts2t} <: AbstractDims
  S::Int
  Fθ2name::TFθ2name
  s2t::Ts2t

end

MeanModelDims(;S, Fθ2name = Dict{Symbol, Any}(),   s2t = collect(1:S), kwargs...) =  MeanModelDims( S, Fθ2name, s2t)
Dims(dims::MeanModelDims; kwargs...) = MeanModelDims(dims; kwargs...)
MeanModelDims(dims::AbstractDims; 
  S=dims.S,
  Fθ2name=dims.Fθ2name,
  kwargs...) = MeanModelDims(;S, Fθ2name)


truncate(::MeanModelDims; smax,kwargs...)=MeanModelDims(;S=smax,Fθ2name=nothing)

#data
struct MeanModelData{
  Tcontainsdata <: ContainsData, 
  Ty<:AbstractVector{Float64}} <: AbstractData{Tcontainsdata}
  y::Ty

  MeanModelData(y::Ty)  where Ty<: AbstractVector{Float64} = (
    new{ifelse(!isempty(y),NoData,HasData),Ty}(y))
end
MeanModelData(;y, kwargs...) = MeanModelData(y)
MeanModelData(data::AbstractData) = MeanModelData(data.y)

truncate(data::MeanModelData; smax,kwargs...)=MeanModelData(;y=data.y[1:smax],)



function loadmeanmodelpriors(::Type{TΘ};data=nothing, y=data.y,) where TΘ <: AbstractModelParameters

  #mostly empirical priors. We model an iid normal distribution
  μ0 = mean(y)
  τμ0 = length(y)/var(y)

  σ0 = std(y)
  τmult=0.9

  #because we marginalize out precision, we don't compute the parameter draws.
  #However, we do need the associated priors

  #from the IG distribution, ζ0/(α0-1)=σ0^2 (set from sd)
  #from the G distribution, E(τ)M=α0/ζ0*M=(α0/ζ0^2)^0.5 (set as sd as mult of prec)
  #therefore:
  α0=1/τmult^2
  ζ0=(α0-1)*σ0^2
  @assert σ0 ≈ (ζ0/(α0-1))^0.5
  @assert (α0/ζ0^2)^0.5 ≈ τmult * α0/ζ0

  hypert = (;
    μ=(; μ0, τμ0),
    τ=(;α0, ζ0)
  )

  return TΘ(;hypert...)
end

#construct the mean model from an existing DGP object
function MeanModel(dgp::AbstractDGP, ::Type{TΘ}=DGPMeanModelParameters;
    numburnrecords=PARAM[:livenumburnrecords], ) where TΘ <: AbstractModelParameters

  dims=MeanModelDims(dgp.dims)
  data=MeanModelData(dgp.data)
  Θ = TΘ(;μ = mean(data.y), τ=1/var(data.y))
  hyper = loadmeanmodelpriors(DGPMeanModelParameters;y=data.y)


  #look-up the number of burn records in case the dgp is already burnt
  #but if it isn't the lookup should return the same value
  @assert dgp.records.burnt || (numburnrecords ≡ dgp.records.numburnrecords)

  records=ChainRecords([:μ, :τ]; 
    Θ, 
    chainid=Symbol(dgp.records.chainid),  
    numsamplerecords=dgp.records.numsamplerecords, 
    numchains=dgp.records.numchains, 
    numburnrecords)

  return DGP(; dims, data, Θ, hyper, records)
end


#WARNING WARNING WARNING- we must draw τ before μ in order to get independent draws
function updateθpτ(dgp::AbstractDGP{<:DGPMeanModelParameters}; )
  @unpack data, Θ, dims, hyper= dgp
  @unpack S = dims
  @unpack μ0,τμ0,α0,ζ0 = hyper
  @unpack y = data

  α=S/2 + α0
  ζ =0.5*(y⋅y) + τμ0/2*μ0^2-(sum(y) + τμ0*μ0)^2/(2*(S+τμ0)) + ζ0

  return (;α,ζ)
end
draw(::Val{:τ}, drawdims...; α, ζ) = rand(Gamma(α, 1/ζ),drawdims...) 


#WARNING WARNING WARNING- we must draw τ before μ in order to get independent draws
function updateθpμ(dgp::AbstractDGP{<:DGPMeanModelParameters}; )
  @unpack data, Θ, dims, hyper= dgp
  @unpack S = dims
  @unpack τ=Θ
  @unpack μ0,τμ0  = hyper
  @unpack y = data

  τμ=τ*(S+τμ0)
  μμ=(sum(y)+τμ0*μ0)/τμ*τ

  return (;τμ,μμ)
end
draw(::Val{:μ}, drawdims...; μμ, τμ) = rand(Normal(μμ, τμ^(-0.5)),drawdims...) 


#This is a function primarily for testing purposes that uses the built-in distribution functions to compute
#the log pdf
#finish the below to check the mean model
function lpdist( dgp::DGP{<:DGPMeanModelParameters}, s::Symbol=:lpost)
  
  #######total posterior
  (s ≡ :post) && return (lpdist(dgp, :ylike)+lpdist(dgp, :μprior)+lpdist(dgp, :τprior)  )

  (s ≡ :yF_llike) && return lpdist(dgp, :ylike)

  @unpack data, Θ, dims,hyper = dgp
  @unpack α0, ζ0, μ0, τμ0 = hyper
  @unpack μ, τ = Θ
  @unpack y = data
  @unpack S=dims



  ######individual priors and likelihoods
  (s ≡ :ylike) && return logpdf(MultivariateNormal(fill(μ,S), I(dims.S)/τ),data.y)
  (s ≡ :μprior) && return logpdf(Normal(μ0, (τ*τμ0)^(-0.5)),μ)
  (s ≡ :τprior) && return logpdf(Gamma(α0, 1/ζ0),τ)



  throw("unrecognized lpdist call $s")
end


#specific conditional draw function for the mean model
function conditionaldraw(θ::Symbol, dgp::DGP{<:DGPMeanModelParameters},)
  if θ ≡ :τ

    @unpack α,ζ = updateθpτ(dgp)
    τ = draw(:τ; α,ζ)
    return DGP(dgp; τ)

  elseif θ ≡ :μ
    @unpack μμ, τμ = updateθpμ(dgp)
    μ = draw(:μ; μμ, τμ)
    return DGP(dgp; μ)
  else
    throw("unrecognized mean model parameter $θ")
  end

  @assert false
end

gettapedeck(::Val, ::DGP{<:DGPMeanModelParameters}) = ()->[:τ, :μ]
  







