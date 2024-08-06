
#inverts a pd matrix always (note there is no tolerance here for singular matrices!)
function pdinv(Λ::Symmetric; verbose=true)
  
  try
    return Λ |> cholesky |> inv |> Symmetric
  catch err
    verbose && @warn("Inversion of pos-def matrix failed with error $err. Attempting SVD")
    #try
      return Λ |> svd |> inv |> Symmetric
    #=catch err
      verbose && @warn("Second attempt of inversion of pos-def matrix failed with error $err. Attempting new SVD algorithm.")
      return svd(Λ, alg = LinearAlgebra.QRIteration()) |> inv
    end=#
  end
end

pdinv(Λ::AbstractMatrix) = Λ |> Symmetric |> pdinv


#collapse(nts;propnames=propertynames(nts)) = collapse(nts,propnames |> Tuple)

#collapses an array of tuples into a tuple of arrays
#using DataFrames leads to significant performance improvements over naive methods
collapse(ds::AbstractArray{<:AbstractDict};) = collapse(ds; colnames=keys(ds[begin])|> collect)
collapse(::AbstractArray{NamedTuple{(), Tuple{}}}; kwargs...) = (;)
function collapse(nts::AbstractVector;colnames = nts[1] |> propertynames)
  colformdf=nts |> DataFrame 
  

  return NamedTuple{colnames}(colformdf |> eachcol |> collect)
end

function collapse(nts::AbstractArray;colnames = nts[1] |> propertynames)
  colformdfs=(nts |> eachcol) .|> DataFrame 
  @eval Main nts=$nts
  @eval Main colnames=$colnames
  mats = map(colnames) do colname
    reduce(hcat, colformdfs .|> colformdf->colformdf[!, colname])
  end
  

  return NamedTuple{colnames |> Tuple}(mats)
end

#collapses keyed collections
function collapse(dicts::AbstractVector{TD}; strict=true) where TD<:AbstractDict
  if strict || allequal(dicts .|> keys)
    @assert allequal(dicts .|> keys)
    return TD(k=>(d->d[k]).(dicts) for k ∈ keys(dicts[begin]))
  end

  allkeys = reduce(vcat, dicts .|> keys .|> collect)
  return TD(k=>(d->haskey(d,k) ? d[k] : missing).(dicts) for k ∈ allkeys)
end


#some utilities around types
fieldandtype(obj::Tobj)  where Tobj = (obj, Tobj)
fieldandtype(Θ, f) = getfield(Θ,f) |> fieldandtype


@inline splattuple(x::Tuple) = x
@inline splattuple(x::Tuple,y::Tuple) = (x...,y...)
@inline splattuple(x::Tuple,y::Tuple, args...) = splattuple((x...,y...), args...)


vecmaybe(x::Real) = x
vecmaybe(v::AbstractVector) = v
vecmaybe(a) = vec(a)

function pickaseed!()
  aseed = rand(1:10^5)
  @info "seed: $aseed"
  Random.seed!(aseed)
end



############binary wrappers
function IN_BIN_STREAM(p::String)
  local obj
  open(ZstdDecompressorStream, p) do io
      obj = deserialize(io)
  end

  return obj
end


const BIN_EXTENSION = "bin.zstd"
function OUT_BIN_STREAM(p::String, obj, level=1)

  io = ZstdCompressorStream(open(p, "w"), level=level)
  try
    serialize(io, obj)
  finally
    close(io)
  end
end





########utility functions relating to file IO#########
#names should be lower case and standard chaacters/numbers
function standardizename(s::AbstractString; keepchars="", onlylowercase=true)
  if onlylowercase
    s = lowercase(s)
  end
  
  return replace(s,Regex("[^a-zA-Z0-9$keepchars]")=>"")
end


#below hack is for handling 2 digit years
function fix2digityear(dt::Date; cutoff=Date(1940,1,1))
  y,m,d = year(dt), month(dt), day(dt)
  if y ≤ 40
    return Date(y + 2000, m, d)
  elseif y ≤ 100
    return Date(y+1900, m, d)
  end
  return dt
end

  #general function to validate dates in a time series
#dates should already be unique and sorted, and potentially gap-less
function validatedates(dates::Vector{Date};
  frequency::NSymbol, 
  validateexactfrequency::Bool=true)

   @assert allunique(dates)
  @assert issorted(dates)

  (frequency === nothing) && return true

  #not much to do with daily data, we already checked the data was unique
  if Symbol(lowercase(string(frequency))) == :day
    (!validateexactfrequency) || throw("Exact validation of daily frequency not supported")
    return true

  #weekly data should have a minimum frequency of 7 days
  elseif Symbol(lowercase(string(frequency))) == :week
    dateseow = (lastdayofweek).(dates)

    if !allunique(dateseow)
      nonuniquedateindices = nonunique(DataFrame(datesyw=dateseow))
      nonuniquedates = dates[nonuniquedateindices]
      throw("weeks and years of weekly data are not unique. Nonunique dates:
        $(collect(zip(dates, dateseow))[(d->d∈nonuniquedates).(dates)])")
    end
    if validateexactfrequency #make sure there are no missing or misaligned weeks
      validdates= firstdayofweek(minimum(dates)):Week(1):lastdayofweek(maximum(dates))
      validdateseow = lastdayofweek.(validdates)
      mismatches = dateseow .≠ validdateseow
      sum(mismatches)==0 || throw(
        "Failed to validate weekly dates with $(sum(mismatches)) mismatches.
        List of mismatches (given, minimum(dates):Week(1):maximum(dates)):
          $(collect(zip(dateseow, validdateseow))[mismatches])")
    end

  #check monthly date frequencies
  elseif Symbol(lowercase(string(frequency))) == :month
    #simple year-month format
    datesym = (d->year(d) .+ month(d) ./ 100).(dates)
    allunique(datesym) || throw("months and years of monthly data are not unique")

    if validateexactfrequency #we don't insist on equality of dates, just year-month
      validdates= firstdayofmonth(minimum(dates)):Month(1):lastdayofmonth(maximum(dates))
      validdatesym = (d->year(d) .+ month(d) ./ 100).(validdates)
      try
        mismatches = datesym .≠ validdatesym
        sum(mismatches)==0 || throw(error("
          Failed to validate monthly dates with $(sum(mismatches)) mismatches.
          List of mismatches (given, minimum(dates):Month(1):maximum(dates)):
            $(collect(zip(datesym, validdatesym))[mismatches])"))
      catch err
        @eval Main datesym = $datesym
        @eval Main validdatesym = $validdatesym
        debugdf = vcat(DataFrame(dates=dates, datesym=datesym),
          DataFrame(validdates=validdates, validdatesym=validdatesym), cols=:union)
        debugdf |> CSV.write("$(PARAM[:testpath])/validdatesym_dump.csv")
        throw(err)
      end
    end

  #can do many checks of quarterly data
  elseif Symbol(lowercase(string(frequency))) == :quarter
    datesyq = (d->year(d) .+ quarterofyear(d) ./ 10).(dates)
    allunique(datesyq) || throw("months and years of monthly data are not unique")

    if validateexactfrequency #we don't insist on equality of dates, just the year-month
      validdatesq= firstdayofquarter(minimum(dates)):Quarter(1):lastdayofquarter(maximum(dates))
      validdatesyq = (d->year(d) .+ quarterofyear(d) ./ 10).(validdatesq)
      try
        mismatchesyq = datesyq .≠ validdatesyq
        sum(mismatchesyq)==0 || throw(error("
          Failed to validate quarterly dates with $(sum(mismatchesyq)) mismatches.
          List of mismatches (given, minimum(dates):Quarter(1):maximum(dates)):
            $(collect(zip(datesyq, validdatesyq))[mismatches])"))
      catch err 
        debugdf = vcat(DataFrame(dates=dates, datesyq=datesyq),
          DataFrame(validdatesq=validdatesq, validdatesyq=validdatesyq), cols=:union)
        debugdf |> CSV.write("$(PARAM[:testpath])/validdatesyq_dump.csv")
        throw(err)
      end

      #each quarter end should fall on the same month
      datesym = (d->year(d) .+ month(d) ./ 100).(dates)
      validdatesm= firstdayofmonth(minimum(dates)):Month(3):lastdayofmonth(maximum(dates))
      validdatesym = (d->year(d) .+ month(d) ./ 100).(validdatesm)
      try
        mismatchesym = datesym .≠ validdatesym
        sum(mismatchesym)==0 || throw(error("
          Failed to validate monthly dates with $(sum(mismatchesym)) mismatches.
          List of mismatches (given, minimum(dates):Month(3):maximum(dates)):
            $(collect(zip(datesym, validdatesym))[mismatchesym])"))
      catch err
        debugdf = vcat(DataFrame(dates=dates, datesym=datesym),
          DataFrame(validdatesm=validdatesm, validdatesym=validdatesym), cols=:union)
        debugdf |> CSV.write("$(PARAM[:testpath])/validdatesym_for_q_dump.csv")
        throw(err)
      end
    end

  elseif Symbol(lowercase(string(frequency))) == :year
    throw("validatedates not (yet) implemented for year frequency.")
  else
    throw("unrecognized date frequency $frequency")
  end
  return nothing
end



#several utilities for drawing from a truncated multivariate normal
#starts with accept/reject, then uses Gibbs if necessary
#expensive but probably cheaper than accept/reject
function truncatedmvn(μ::AbstractVector, Σ; 
    lower=-Inf, 
    upper=Inf, 
    Λ,
    maxiter=10^3, #1000 tries to get a cheap answer
    usefallback=true)


  for i in 1:maxiter
    xcand = rand(MultivariateNormal(μ, Σ))
    all(lower .≤ xcand .≤ upper) && return xcand
  end

  #fallback is expensive and somewhat approximate but always provides valid vlaues
  usefallback && return truncatedmvngibbs(μ,; lower, upper, Λ)
  throw("Unable to find draw that fulfills truncation constraints")
end

#See Gabriel Rodriguez-Yam, Richard A. Davis, and Louis L. Scharf 2004
function truncatedmvngibbs(μ::AbstractVector, Σ=nothing; 
    lower, 
    upper, 
    burnin=100,
    Λ = m.pdinv(Σ))

  K= length(μ)

  #start by a simple univariate draw

  σ2ck = [inv(Λ[k,k]) for k ∈ 1:K]
  τkNk = [@view(Λ[Not(k),k]) for k ∈ 1:K]

  #initial draw
  x = map(1:K) do k
    rand(Truncated(Normal(μ[k], σ2ck[k]^0.5), lower, upper))
  end

  #draw from the conditional
  for i ∈ 1:burnin
    for k ∈ 1:K
      μc = μ[k] - σ2ck[k]*τkNk[k]'*(@view(x[Not(k)])-@view(μ[Not(k)]))
      x[k] = rand(Truncated(Normal(μc, σ2ck[k]^0.5), lower, upper, ))
    end
  end

  return x
end

#subtract off the max log weight for numerical stability
function stableweights(;lw̃)
  maxlw̃ = maximum(lw̃)
  lw̃stable = lw̃ .- maxlw̃

  return (exp.(lw̃stable))./sum(exp.(lw̃stable))
end

function teststableweights()
  lw̃  = rand(Laplace(),10^4)
  wact = (lw̃ .|> exp) |> w̃->w̃./sum(w̃)
  w = stableweights(;lw̃)

  @assert all((w .≈ wact) .| ((w .+ 1.0) .≈ (wact .+ 1.0)))

  @info "Passed test of stableweights"
end


#provides the gamma parameters form the moments
function momgamma(;μ,σ2,testmomgamma=false)
  α = μ^2/σ2
  ζ = μ/σ2


  @assert mean(Gamma(α,1/ζ)) ≈ μ
  @assert var(Gamma(α,1/ζ)) ≈ σ2

  return (α,ζ)
end

#this perhaps should be used for the man priors on ν
function matchtruncatedgammamoments(;lower::Real, μ, σ2, 
    verbose=false, 
    optkwargs=(;))
  @assert lower>0 && μ>0 && σ2 > 0

  #these are from mathematica- see gammacalcs.nb and/or the model doc
  tgammaμ(α,ζ)=exp(loggamma(1+α,lower*ζ)-(log(ζ) + loggamma(α)))
  tgammaσ2(α,ζ, μ=tgammaμ(α,ζ))=exp(loggamma(2+α,lower*ζ)-(2*log(ζ) + loggamma(α))) - μ^2
  function obj(α,ζ)
    μ̂ = tgammaμ(α,ζ)
    σ̂2=tgammaσ2(α,ζ)

    return (μ-μ̂)^2+(σ2-σ̂2)^2
  end

  obj(Θ)=obj(Θ[1],Θ[2])

  #set the initial values to the parameters implied for the untruncated gamma
  Θ0 = [μ^2/σ2,μ/σ2]
  opt = optimize(obj, Θ0, NelderMead(), ;optkwargs...)

  α0,ζ0 = Optim.minimizer(opt)
  flag=Optim.converged(opt)
  if verbose
    simgammat=rand(Truncated(Gamma(α0,1/ζ0),lower,Inf),10^5)
    μimplied, _ = quadgk(x->x*pdf(Gamma(α0,1/ζ0),x),lower, Inf)
    σ2implied, _ = quadgk(x->(x-μimplied)^2*pdf(Gamma(α0,1/ζ0),x),lower,Inf)
    @info "α0=$α0,ζ0=$ζ0
      convergence status: $flag
      implied μ=$μimplied (act=$μ), implied σ2=$(σ2implied) (act=$σ2)

      $opt"
  end

  return (;α0,ζ0,flag)
end

#approximately less than or equal to
⪅(a::Float64,b::Float64) = a-eps(a)^0.5 ≤ b+eps(b)^0.5
⪅(a::Float64,b::Int) = a-eps(a)^0.5 ≤ b
⪅(a::Int,b::Int) = a ≤ b
⪅(a::Int,b::Float64) = a ≤ b+eps(b)^0.5
⪅(::Missing,b) = missing
⪅(a,::Missing) = missing
⪅(a,b) = ⪅(Float64(a),Float64(b))
⪊(a,b) = ⪅(b,a)
