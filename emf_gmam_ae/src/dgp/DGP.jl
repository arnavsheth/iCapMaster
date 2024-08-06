

##############################
abstract type ContainsData end
struct HasData <: ContainsData end
struct NoData <: ContainsData end

@kwdef struct Data{
    Tcontainsdata<:ContainsData, 
    Ty<:AbstractVector{Float64}, 
    TF<:AbstractMatrix{Float64}, 
    Tr<:AbstractVector{Float64}} <: AbstractData{Tcontainsdata}
  y::Ty
  F::TF
  r::Tr

  function Data(y::Ty,F::TF,r::Tr) where {
    Ty<:AbstractVector{Float64}, 
    TF<:AbstractMatrix{Float64}, 
    Tr<:AbstractVector{Float64}}

    @assert size(F,1) == length(r)
    @assert isempty(y) == isempty(F) == isempty(r)

    return new{ifelse(isempty(y),NoData,HasData), Ty, TF, Tr}(y,F,r)
  end
end


@kwdef struct FrData{
  Tcontainsdata <: ContainsData, 
  TF<:AbstractMatrix{Float64}, 
  Tr<:AbstractVector{Float64}} <: AbstractData{Tcontainsdata}

  F::TF
  r::Tr

  function FrData(F::TF,r::Tr) where {
    TF<:AbstractMatrix{Float64}, 
    Tr<:AbstractVector{Float64}}
    @assert size(F,1) == length(r)
    @assert isempty(F) == isempty(r)
    return new{ifelse(isempty(F),NoData,HasData), TF, Tr}(F,r)
  end
end  


#truncates the data for CV purposes
function truncate(data::AbstractData; smax,dims)
  @unpack s2t = dims
  y = data.y[1:smax]
  F = data.F[1:s2t[smax], :]
  r = data.r[1:s2t[smax]]

  return Data(;y,F,r)
end



######################################
#A DGP consists of hyperparameters hyper, data D, dimensions dims, and parameters Θ, and a 
#for VB, Θ will consist of moments. For MCMC, values/vectors
struct DGP{
    TΘ<:AbstractModelParameters,
    Tdata<:AbstractData, 
    Thyper<:AbstractModelParameters, 
    Trecords<:Union{ChainRecords, Nothing},
    Tdims<:Union{AbstractDims, NamedTuple}} <: AbstractDGP{TΘ}

  Θ::TΘ #stately expectations
  data::Tdata #immutable data
  hyper::Thyper #immutable hyperparameters
  records::Trecords
  dims::Tdims #important problem dimensions and other structural hyper parameters
end

#the key-word version performs some basic checks
function DGP(;Θ,data,hyper,records, dims)
  @assert allunique(propertynames(Θ)) "Non-unique parameter names found- make sure all properties and sub-properties are unique"
  @assert allunique(propertynames(hyper)) "Non-unique hyperparameter names found- make sure all properties and sub-properties are unique"

  return DGP(Θ,data,hyper,records,dims)

end

#update a DGP recycling old data where appropriate
function DGP(old::Told;strict=true, copyΘ=false, records=old.records, params...)  where Told<:AbstractDGP
  @unpack data,hyper,dims = old

  Θ = (copyΘ ? old.Θ : deepcopy(old.Θ))(;strict, params...)

  dgp = DGP(; data, hyper, dims, Θ, records)

  if strict && typeof(dgp) != Told
    @error "DGP updated to a different type
    ***TΘ
      old=$(typeof(old.Θ))

      new=$(typeof(dgp.Θ))

    ***records
      old=$(typeof(old.records))

      new=$(typeof(dgp.records))    
    "

    throw("DGP updated to different type")
  end

  return dgp
end

DGP(old::AbstractDGP, Θ::AbstractModelParameters) = throw(
  "interestingly, this was never set up properly.")
#DGP(old::AbstractDGP, Θ::AbstractModelParameters) = DGP(;
#     data=old.data, hyper=old.hyper, dims=old.dims, Θ, records=old.records)

#truncates a dgp object where appropriate- useful for CV
#usually we don't want the records object, the exception being "true" CV
function truncate(dgp::AbstractDGP, records::Trecords=nothing; smax, 
  dims=truncate(dgp.dims; smax),
  Θ = truncate(dgp.Θ; smax, dims),
  data = truncate(dgp.data; smax, dims),
  numburnrecords=nothing,
  numsamplerecords=nothing,
  hyper=dgp.hyper,
  ) where Trecords
  
  Trecords <: Union{Nothing, ChainRecords} && return DGP(;Θ, data,hyper,records, dims)
  records == :old && return DGP(;Θ, data,hyper,records=dgp.records, dims)
  
  if records == :newrecords
    @assert numburnrecords !== nothing

    newrecords = empty(dgp.records; Θ, numburnrecords,numsamplerecords=something(numsamplerecords, dgp.records.numsamplerecords))
    return DGP(;Θ, data,hyper,records=newrecords, dims)
  end
  
  



  throw("invalid records argument")
end


function prioronlydgp(dgp::AbstractDGP{<:AbstractModelParametersG};
    mcmcinitmethod=:nodata,
    fieldstocapture=[:ϕ,:β,:γ,:ω,:ν,:τy,:τx,:τβ,:τϕ])

  data0=m.Data(;y=Float64[], F=reshape(Float64[],0,dgp.dims.K), r=Float64[])
  @assert typeof(data0) <: m.AbstractData{<:m.NoData}

  dims0 = m.Dims(dgp.dims; S=0, dates=Date[])
  Θ0 = initmcmc(Val(mcmcinitmethod), typeof(dgp.Θ); data=data0, dims=dims0, hyper=dgp.hyper)

  records0=empty(dgp.records; Θ=Θ0, fieldstocapture)

  dgp0=m.DGP(;
    hyper=dgp.hyper, 
    records=records0, 
    dims=dims0, 
    data=data0,
    Θ = Θ0)
  return dgp0
end

#This is a function primarily for testing purposes that uses the built-in distribution functions to compute
#the log pdf
function lpdist( dgp::AbstractDGP, s::Symbol=:post) 
  
  #######total posterior
  (s ≡ :post) && return (lpdist(dgp, :ylike)+lpdist(dgp, :xprior)+lpdist(dgp, :ϕprior)
    +lpdist(dgp, :βprior)+lpdist(dgp, :γprior)+lpdist(dgp, :ωprior)
    +lpdist(dgp, :τyprior)+lpdist(dgp, :τxprior)+lpdist(dgp, :ψprior)+lpdist(dgp, :νprior))
  (s ≡ :yF_llike) && return (lpdist(dgp, :ylike) + lpdist(dgp, :xprior))


  @unpack data, Θ, dims,hyper = dgp
  @unpack ϕ0, β0, βΔ0, A0, M0, v, κ0, δ0, αy0, ζy0, αx0, ζx0, αν0,ζν0, νmin, νmax = hyper
  @unpack x, ϕ, β, γ, ω, τx, τy, ψ, ν = Θ
  @unpack y, F = data

  ######individual priors and likelihoods
  (s ≡ :ylike) && return logpdf(MultivariateNormal(formX̃L(;x, dims)*ϕ + formxS(x; dims), I(dims.S)/τy),data.y)
  (s ≡ :xprior) && return logpdf(MultivariateNormal(data.F * β+data.r, inv(formΨ(ψ)) ./ (τx*τy) ),x)
  (s ≡ :ϕprior ) && return logpdf(MultivariateNormal(ϕ0, pdinv(M0) ./ τy ),ϕ)
  (s ≡ :γprior) && return sum(γ .|> γk->logpdf(Bernoulli(ω), γk))
  (s ≡ :ωprior) && return logpdf(Beta(κ0, δ0), ω)
  (s ≡ :τyprior) && return logpdf(Gamma(αy0, 1/ζy0), τy)
  (s ≡ :τxprior) && return logpdf(Gamma(αx0, 1/ζx0), τx)
  (s ≡ :ψprior) && return sum(ψ .|> ψt->logpdf(Gamma(ν/2, 1/(ν/2)), ψt))
  #(s ≡ :νprior) && return logpdf(Uniform(νmin, νmax), ν)
  (s ≡ :νprior) && return isfinite(νmax-νmin) ? logpdf(Truncated(Gamma(αν0,1/ζν0), νmin, νmax), ν) : logpdf(Gamma(αν0,1/ζν0),ν)
  if s ≡ :βprior #βprior is a bit more complex than the others
    D = formD(γ,v)
    Dinv = D\I
 
    #recall multiplying Dinv*β0 dramatically simplifies the math for having non-zero β priors
    return logpdf(MultivariateNormal(β0+Dinv*βΔ0, Symmetric(inv(D*A0*D)) ./ (τx*τy) ),β)
  end




  throw("unrecognized lpdist call $s")
end

#This is a function primarily for testing purposes that uses the built-in distribution functions to compute
#the log pdf
function lpdist( dgp::DGP{<:DGPModelParametersG}, s::Symbol=:post)
  
  #######total posterior
  (s ≡ :post) && return (lpdist(dgp, :ylike)+lpdist(dgp, :xprior)+lpdist(dgp, :ϕprior)
    +lpdist(dgp, :βprior)+lpdist(dgp, :γprior)+lpdist(dgp, :ωprior)
    +lpdist(dgp, :τyprior)+lpdist(dgp, :τxprior)+lpdist(dgp, :ψprior)+lpdist(dgp, :νprior)
    +lpdist(dgp, :τϕprior)+lpdist(dgp, :τβprior))
  
  (s ≡ :yF_llike) && return (lpdist(dgp, :ylike) + lpdist(dgp, :xprior))


  @unpack data, Θ, dims,hyper = dgp
  @unpack ϕ0, β0, βΔ0, A0, M0, v, κ0, δ0, αy0, ζy0 = hyper
  @unpack αϕ0, ζϕ0, αβ0, ζβ0, αx0, ζx0, αν0,ζν0, νmin, νmax = hyper
  @unpack x, ϕ, β, γ, ω, τx, τy, ψ, ν, τβ,τϕ = Θ
  @unpack y, F = data

  ######individual priors and likelihoods
  (s ≡ :ylike) && return logpdf(MultivariateNormal(formX̃L(;x, dims)*ϕ + formxS(x; dims), I(dims.S)/τy),data.y)
  (s ≡ :xprior) && return logpdf(MultivariateNormal(data.F * β+data.r, inv(formΨ(ψ)) ./ (τx*τy) ),x)
  (s ≡ :ϕprior ) && return logpdf(MultivariateNormal(ϕ0, pdinv(M0) .* (1/ τy / τϕ )),ϕ)
  (s ≡ :γprior) && return sum(γ .|> γk->logpdf(Bernoulli(ω), γk))
  (s ≡ :ωprior) && return logpdf(Beta(κ0, δ0), ω)
  (s ≡ :τyprior) && return logpdf(Gamma(αy0, 1/ζy0), τy)
  (s ≡ :τxprior) && return logpdf(Gamma(αx0, 1/ζx0), τx)
  (s ≡ :τϕprior) && return logpdf(Gamma(αϕ0, 1/ζϕ0), τϕ)
  (s ≡ :τβprior) && return logpdf(Gamma(αβ0, 1/ζβ0), τβ)  
  (s ≡ :ψprior) && return sum(ψ .|> ψt->logpdf(Gamma(ν/2, 1/(ν/2)), ψt))
  #(s ≡ :νprior) && return logpdf(Uniform(νmin, νmax), ν)
  (s ≡ :νprior) && return isfinite(νmax-νmin) ? logpdf(Truncated(Gamma(αν0,1/ζν0), νmin, νmax), ν) : logpdf(Gamma(αν0,1/ζν0),ν)
  if s ≡ :βprior #βprior is a bit more complex than the others
    D = formD(γ,v)
    Dinv = D\I
 
    #recall multiplying Dinv*β0 dramatically simplifies the math for having non-zero β priors
    return logpdf(MultivariateNormal(β0 + Dinv*βΔ0, Symmetric(inv(D*A0*D)) .* (1 / (τβ*τx*τy)) ),β)
  end




  throw("unrecognized lpdist call $s")
end



#This is a function primarily for testing purposes that uses the built-in distribution functions to compute
#the log pdf
function lpdist( dgp::DGP{<:DGPModelParametersGIR}, s::Symbol=:post)
  
  #######total posterior
  (s ≡ :post) && return (lpdist(dgp, :ylike)+lpdist(dgp, :xprior)+lpdist(dgp, :ϕprior)
    +lpdist(dgp, :βprior)+lpdist(dgp, :γprior)+lpdist(dgp, :ωprior)
    +lpdist(dgp, :τyprior)+lpdist(dgp, :τxprior)+lpdist(dgp, :ψprior)+lpdist(dgp, :νprior)
    +lpdist(dgp, :τϕprior)+lpdist(dgp, :τβprior))

    (s ≡ :yF_llike) && return (lpdist(dgp, :ylike) + lpdist(dgp, :xprior))

  @unpack data, Θ, dims,hyper = dgp
  @unpack T = dims
  @unpack ϕ0, β0, βΔ0, A0, M0, v, κ0, δ0, αy0, ζy0 = hyper
  @unpack αϕ0, ζϕ0, αβ0, ζβ0, αx0, ζx0, αν0,ζν0, νmin, νmax = hyper
  @unpack x, ϕ, β, γ, ω, τx, τy, ψ, ν, τβ,τϕ = Θ
  @unpack y, F = data

  ######individual priors and likelihoods
  (s ≡ :ylike) && return logpdf(MultivariateNormal(formX̃L(;x, dims)*ϕ + formxS(x; dims), I(dims.S)/τy),data.y)
  (s ≡ :xprior) && return logpdf(MultivariateNormal(data.F * β+data.r*sqrt(T), inv(formΨ(ψ)) .* T / (τx*τy) ),x*sqrt(T))
  (s ≡ :ϕprior ) && return logpdf(MultivariateNormal(ϕ0, pdinv(M0) .* (1/ τy / τϕ )),ϕ)
  (s ≡ :γprior) && return sum(γ .|> γk->logpdf(Bernoulli(ω), γk))
  (s ≡ :ωprior) && return logpdf(Beta(κ0, δ0), ω)
  (s ≡ :τyprior) && return logpdf(Gamma(αy0, 1/ζy0), τy)
  (s ≡ :τxprior) && return logpdf(Gamma(αx0, 1/ζx0), τx)
  (s ≡ :τϕprior) && return logpdf(Gamma(αϕ0, 1/ζϕ0), τϕ)
  (s ≡ :τβprior) && return logpdf(Gamma(αβ0, 1/ζβ0), τβ)  
  (s ≡ :ψprior) && return sum(ψ .|> ψt->logpdf(Gamma(ν/2, 1/(ν/2)), ψt))
  #(s ≡ :νprior) && return logpdf(Uniform(νmin, νmax), ν)
  (s ≡ :νprior) && return isfinite(νmax-νmin) ? logpdf(Truncated(Gamma(αν0,1/ζν0), νmin, νmax), ν) : logpdf(Gamma(αν0,1/ζν0),ν)
  if s ≡ :βprior #βprior is a bit more complex than the others
    D = formD(γ,v)
    Dinv = D\I
 
    #recall multiplying Dinv*β0 dramatically simplifies the math for having non-zero β priors
    return logpdf(MultivariateNormal(β0 + Dinv*βΔ0, Symmetric(inv(D*A0*D)) .* (1 / (τβ*τx*τy)) ),β)
  end




  throw("unrecognized lpdist call $s")
end

