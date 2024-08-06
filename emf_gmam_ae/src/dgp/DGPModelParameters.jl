######################################
#this is a very generic structure for holding information about the parameters
#to begin with, most of this will be keywords.
@kwdef struct DGPModelParameters{Tx,Tϕ,Tβ,Tγ,Tω,Tτy,Tτx,Tψ,Tν} <: AbstractModelParameters
  x::Tx
  ϕ::Tϕ  
  β::Tβ

  γ::Tγ
  ω::Tω

  τy::Tτy  
  τx::Tτx
  ψ::Tψ
  ν::Tν
end

@kwdef struct DGPModelParametersG{Tx,Tϕ,Tβ,Tγ,Tω,Tτy,Tτx,Tgy, Tgx,Tψ,Tν} <: AbstractModelParametersG
  x::Tx
  ϕ::Tϕ  
  β::Tβ

  γ::Tγ
  ω::Tω

  τy::Tτy  
  τx::Tτx
  τϕ::Tgy
  τβ::Tgx
  ψ::Tψ
  ν::Tν
end


@kwdef struct DGPModelParametersGIR{Tx,Tϕ,Tβ,Tγ,Tω,Tτy,Tτx,Tgy, Tgx,Tψ,Tν} <: AbstractModelParametersG
  x::Tx
  ϕ::Tϕ  
  β::Tβ

  γ::Tγ
  ω::Tω

  τy::Tτy  
  τx::Tτx
  τϕ::Tgy
  τβ::Tgx
  ψ::Tψ
  ν::Tν
end


@kwdef struct DGPModelParametersGT{Tx,Tϕ,Tβ,Tγ,Tω,Tτy,Tτx,Tgy, Tgx,Tψ,Tν} <: AbstractModelParametersG
  x::Tx
  ϕ::Tϕ  
  β::Tβ

  γ::Tγ
  ω::Tω

  τy::Tτy  
  τx::Tτx
  τϕ::Tgy
  τβ::Tgx
  ψ::Tψ
  ν::Tν
end


#this is the parameters for the marginalized mean model
@kwdef struct DGPMeanModelParameters{Tμ,Tτ} <: AbstractModelParameters
  μ::Tμ
  τ::Tτ
end


MODEL_INDEX=Dict(
  :standard=>DGPModelParameters,
  :standardG=>DGPModelParametersG,
  :standardGIR=>DGPModelParametersGIR,
  :standardGT=>DGPModelParametersGT,
  :mean=>DGPMeanModelParameters
)

#a shortcut for getting at components attatched to parameters
function Base.getproperty(Θ::TΘ, p::Symbol) where TΘ <: AbstractModelParameters
  #=shortcut access for fields
  if hasfield(TΘ, p)
    f,Tf = fieldandtype(Θ,p)
    (isprimitivetype(Tf) || (Tf <: AbstractArray)) && return f
    hasfield(Tf, p) && return getfield(f,p)
  end=#
  (hasfield(TΘ, p)) && return getfield(Θ,p)

  
  for fn ∈ fieldnames(TΘ)
    θ = getfield(Θ,fn)
    hasproperty(θ,p) && return getfield(θ,p)
  end

  #try to access by parsing the Symbol- MCMCChains sometimes uses Symbols to access arrays
  #slight performance penalty (~500ns)
  Fgroup = match(r"[^\[]+", p |> string).match |> Symbol
  if hasproperty(Θ, Fgroup)
    return p |> string |>  s->getproperty(Θ,Fgroup)[parse(Int,match(r"[0-9]+",s).match)]
  end

  throw("unrecognized parameter property $p")
end

#more helper methods for getting properties, this time via getindex
Base.getindex(Θ::TΘ, p::Symbol) where TΘ <: AbstractModelParameters = getproperty(Θ,p)
Base.getindex(Θ::TΘ, ps::AbstractVector{Symbol}) where TΘ <: AbstractModelParameters = (p->Θ[p]).(ps)
Base.getindex(Θ::TΘ, args...) where TΘ <: AbstractModelParameters = Θ[args]

isdatavalue(::Type{<:AbstractVector}) = true
isdatavalue(::Type{<:Real}) = true
isdatavalue(::Type{<:AbstractModelParameter}) = false
isdatavalue(::Type{<:NamedTuple}) = false
isdatavalue(::Type{<:Nothing}) = true

#the below is incredibly fast as the entire funciton seems to be inferred at compile time
function Base.propertynames(::TΘ) where TΘ <: AbstractModelParameters
  fnms = fieldnames(TΘ)

  Tfs = fnms .|> fn->fieldtype(TΘ,fn)
  #splattuple is in the Utilities section- just efficiently merges tupels
  nms = splattuple(fnms, reduce(splattuple, Tfs .|> (Tf->isdatavalue(Tf) ? () : fieldnames(Tf))))
  return nms
end


#update a parameters object recycling old data where appropriate
(old::DGPModelParameters)(;kwargs...) = updatemodelparameters(old, DGPModelParameters; kwargs...)
(old::DGPModelParametersG)(;kwargs...) = updatemodelparameters(old, DGPModelParametersG; kwargs...)
(old::DGPModelParametersGIR)(;kwargs...) = updatemodelparameters(old, DGPModelParametersGIR; kwargs...)
(old::DGPModelParametersGT)(;kwargs...) = updatemodelparameters(old, DGPModelParametersGT; kwargs...)
(old::DGPMeanModelParameters)(;kwargs...) = updatemodelparameters(old, DGPMeanModelParameters; kwargs...)
function updatemodelparameters(old::Told, ::Type{Tnew},;strict=true, params...
    ) where {Told <:AbstractModelParameters, Tnew <:AbstractModelParameters}

  isempty(params) && return old
  
  allfields = fieldnames(Told)
  newfields = keys(params)


  #recyle the old data
  uptodate = [p=>(p ∈ newfields ? params[p] : getfield(old, p) ) for p ∈ allfields]
  #@eval Main allfields = $allfields 
  @assert setdiff( keys(params), propertynames(old)) |> isempty "
    Excess keys passed to DGPModelParameters: 
    setdiff( keys(params), propertynames(old) = $(setdiff( keys(params), propertynames(old)))"
  Θ = Tnew(; uptodate...)

  strict && (@assert typeof(Θ) == Told "typeof(Θ) = $(typeof(Θ)) while Told=$Told !!")
  return Θ
end

#truncates the local parameters for CV purposes
function truncate(Θ::AbstractModelParameters; smax, dims)

  @unpack s2t = dims

  x = Θ.x[1:s2t[smax]]
  ψ = Θ.ψ[1:s2t[smax]]

  return Θ(;x,ψ, strict=false)
end
truncate(Θ::DGPMeanModelParameters; kwargs...)=Θ

