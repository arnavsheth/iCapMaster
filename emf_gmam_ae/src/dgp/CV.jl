

#QUESTION- do I want to only assess performance on F, and thereby assess the point P+Δt out, or do I want to 
#assess only a single unit of Δt forward and potentially feed in the preivous calculations of x
#note we could include this as yet another predictive method



#load all cv relevant parameters
#putting this into its own function to facilitate later access
function loadcvparams(dgp=nothing;
  P=dgp.dims.P,
  S=dgp.dims.S,
  Δt = dgp.dims.Δt,
  holdoutmethod=PARAM[:cvholdoutmethod], 
  Fprediction=PARAM[:cvFprediction],
  sstepsaheadmethod::Tsstepsaheadmethod=PARAM[:cvsstepsahead],
  preferrednumtestpoints=PARAM[:cvpreferrednumtestpoints][Δt],
  cvweightingmethod=PARAM[:cvweightingmethod],
  retrainingfrequency=PARAM[:cvretrainingfrequency][Δt],
  holdoutmethodlabel=PARAM[:iocvholdoutmethodlabels][holdoutmethod]
  )  where Tsstepsaheadmethod


  cvparams=Dict()

  if Tsstepsaheadmethod <: Int
    sstepsahead=sstepsaheadmethod
  elseif sstepsaheadmethod==:first
    sstepsahead=1
  elseif sstepsaheadmethod==:nooverlap
    sstepsahead=P÷Δt + 1
  else
    throw("unrecognized sstepsaheadmethod=$sstepsaheadmethod")
  end


  #need to account for some edge cases here- always want to run the analysis, even if its just one point
  cvparams[:smaxstart] = min(max(S - preferrednumtestpoints - sstepsahead + 1,S ÷ 2, 1),S - sstepsahead)
  cvparams[:smaxend] = S - sstepsahead
  cvparams[:Stest] = S - cvparams[:smaxstart] - sstepsahead + 1
  cvparams[:sstepsahead]=sstepsahead
  if !(length(cvparams[:smaxstart]:cvparams[:smaxend]) == cvparams[:Stest])
    @eval Main cvparams=$cvparams
    throw("length(cvparams[:smaxstart]:cvparams[:smaxend]) ≠ cvparams[:Stest]")
  end


  predictionlabel = "$Fprediction" |> str-> Symbol(str[1:min(length(str),5)])
  cvparams[:cvparamgroupname] = "cv$(holdoutmethodlabel)_P$(sstepsahead)"
  cvparams[:cvparamgroupname_pred] = "$(cvparams[:cvparamgroupname])_pred$(predictionlabel)" |> Symbol
  cvparams[:cvparamgroupname_pred_mean] = "$(cvparams[:cvparamgroupname])_mean_pred$predictionlabel" |> Symbol
  cvparams[:cvparamgroupname_err] = "$(cvparams[:cvparamgroupname])_err$(predictionlabel)" |> Symbol
  cvparams[:cvparamgroupname_err_mean] = "$(cvparams[:cvparamgroupname])_mean_err$predictionlabel" |> Symbol

  if cvweightingmethod ≡ :uniform
    cvweights = ones(cvparams[:Stest])
  elseif cvweightingmethod ≡ :linear
    cvweights = 1:cvparams[:Stest] |> collect .|> Float64
  else
    throw("unrecognized CV weighting method")
  end
  cvweights ./= sum(cvweights)
  cvparams[:cvweights] = cvweights


  return (;Fprediction, holdoutmethod, retrainingfrequency, cvparams...)
end


function estimatemeanmodel(dgp::AbstractDGP; 
  mcmcruntype=PARAM[:mcmcruntype],
  Fldensity = PARAM[:cvFldensity],
  numburnrecords,
  )

  @unpack numsamplerecords = dgp.records
  #@eval Main dgp=$dgp
  #@eval Main numburnrecords=numburnrecords

  #use the mean model as a baseline
  dgpmean = MeanModel(dgp; numburnrecords)
  #@eval Main numsamplerecords = $numsamplerecords
  #@eval Main numburnrecords = $numburnrecords
  #@eval Main dgpmean=$dgpmean
  #@eval Main dgp=$dgp

  dgpmean = mcmc(mcmcruntype, dgpmean; 
    maxrecords=numburnrecords+numsamplerecords, 
    stoprule=:bootstrapess, 
    verbose=false)
  #@eval Main dgpmean=$dgpmean

  
  #=densitystats = (integrateposterior(dgpmean; ) do dgpi
    derived(Val{Fldensity}(), dgpi)
  end) |> collapse

  augmentations = Dict(Fstat=>densitystats[Fstat] for Fstat ∈ keys(densitystats))=#

  #augmentedrecords = augmentchainrecords(dgpmean.records, augmentations)
  dgpmean = addderived(Fldensity; dgp=dgpmean)
  #dgpmean = addderived(Fldensity,dgp= dgpmean)
  dgpmean = DGP(dgpmean; records=burnchainrecords(;dgp=dgpmean), strict=false)



  @assert dgpmean.records.numsamplerecords ≡ dgp.records.numsamplerecords
  @assert dgpmean.records.numrecords ≡ dgp.records.numsamplerecords

  return dgpmean
end


function crossvalidatedr2s(dgp::AbstractDGP; 
  cvoverrides=PARAM[:iocvoverrides],
  verbose=PARAM[:cvverbose],
  Fldensity = PARAM[:cvFldensity],
  Fdensity = PARAM[:cvFdensity],
  numburnrecords=nothing,
  skipcvonbase=PARAM[:histpriors][:skipcvonbase],
  priorlinkindex=PARAM[:histpriors][:priorlinkindex])

  @unpack dims, records = dgp

  #@eval Main numburnrecords=$numburnrecords
  #@eval Main dgp=$dgp
  dgpmean = estimatemeanmodel(dgp; numburnrecords)
  #in addition to cvr2 approximations, the densities are used for the Bayes factor
  if  Fldensity ∈ dgp.records.fieldstocapture
    dgp = DGP(dgp; records=augmentchainrecords(dgp.records, Dict(
        Symbol(Fldensity,:_mean)=>dgpmean.records[Fldensity],
        Symbol(Fdensity,:_mean)=>dgpmean.records[Fdensity])), strict=false)
  end


  if (dims.priorset ≡ :histpriors)  && skipcvonbase
    fundid = records.chainid
    (fundid ∈ values(priorlinkindex)) && (!haskey(priorlinkindex, fundid)) && return dgp
  end
      

  for cvoverride ∈ cvoverrides
    cvparams = loadcvparams(dgp; cvoverride...)
    dgp = crossvalidatedr2(dgp; numburnrecords, cvparams, dgpmean, verbose).dgp
  end

  return dgp
end

#primary function for computing CV
#numburnrecords is only needed if we are running true CV
function crossvalidatedr2(dgp::AbstractDGP; 
  verbose=PARAM[:cvverbose],
  cvparams = loadcvparams(dgp; ),
  numburnrecords=nothing,
  dgpmean
  )



  @unpack dims, Θ, records, hyper, data = dgp
  @unpack numsamplerecords,numchains, numrecords = records
  @unpack S,T, s2t, P, Δt, Fθ2name = dims
  @unpack y = data
  @assert records.burnt



  @unpack Fprediction, holdoutmethod, sstepsahead, cvweights, smaxstart, smaxend, Stest, retrainingfrequency  = cvparams
  @unpack cvparamgroupname_pred, cvparamgroupname_pred_mean, cvparamgroupname_err, cvparamgroupname_err_mean =cvparams


  if !isempty(setdiff(propertynames(Θ), records.fieldstocapture))
    @warn "Fields $(setdiff(propertynames(Θ), records.fieldstocapture)) not found in mcmc records.
    Skipping CV analysis."
    return (;dgp)
  end

  if !haskey(dgp.records.chainfieldindex, Fprediction)
    @warn "$Fprediction not found in records, skipping CV"
    return (;dgp)
  end

  cvlabels = Dict()


  @info "Running CV for Fprediction=$Fprediction, holdoutmethod $holdoutmethod, stepsahead=$sstepsahead"

  #compute the expected squared loss for both the estimated and the mean model

  ses = Array{Float64,3}[]
  ŷs = Array{Float64,3}[]
  sizehint!(ŷs, Stest)
  sizehint!(ses, Stest)

  #three types of dgp- 
  # dgpbase contains the prediction and likelihood denominator used in importance sampling methods
  # dgpfull is the full sample dgp
  # dgps is the truncated dgp created by the holdout method, used to either predict or 
  # weight the predictions of a dgpbase.
  dgpbase = dgp
  for smax ∈ smaxstart:smaxend

    @assert smax ≤ S - sstepsahead
    Fpredictions = Symbol(Fprediction, "[$(smax+sstepsahead)]")

    sizepredictions = size(dgp.records[Fpredictions])
    @assert ((sizepredictions ≡ (numsamplerecords, 1, numchains)) || 
      (sizepredictions ≡ (numsamplerecords,numchains)) || 
      ((sizepredictions ≡ (numsamplerecords,)) && numchains==1)) "
      inconsistent size of prediction: size(dgp.records[Fprediction]) = $(size(dgp.records[Fprediction]))"

    #match to the correct date or entry
    cvlabel = haskey(Fθ2name, Fpredictions) ? Fθ2name[Fpredictions] : Fpredictions
    cvlabels[Symbol(cvparamgroupname_pred, "[$(smax-smaxstart+1)]")]= cvlabel
    cvlabels[Symbol(cvparamgroupname_err, "[$(smax-smaxstart+1)]")]= cvlabel

    #get the standard errors
    #the base dgp may be updated if the holdout method is adaptive
    @unpack ŷ, se, newdgpbase = squarederror(Val{holdoutmethod}(); 
      dgpbase, 
      dgpfull=dgp,
      smax, 
      Fprediction=Fpredictions, 
      sstepsahead, 
      verbose, 
      numburnrecords,
      retrainingfrequency)

    #holdoutmethodlabel="$holdoutmethod"
    #@eval Main holdoutmethod=$holdoutmethodlabel
    #@info "holdoutmethod=$holdoutmethod $(size(se))"
    #@eval Main se=$se
    @assert size(se) ≡ (numrecords, numchains)
    @assert size(ŷ) ≡ (numrecords, numchains)
    push!(ses, reshape(se, numrecords, 1, numchains))
    push!(ŷs, reshape(ŷ, numrecords, 1, numchains))
    dgpbase = something(newdgpbase, dgpbase)
  end

  dgpmeanbase=dgpmean
  ŷs_mean = Array{Float64,3}[]
  ses_mean = Array{Float64,3}[]
  sizehint!(ŷs_mean, Stest)
  sizehint!(ses_mean, Stest)
  for smax ∈ smaxstart:smaxend

    Fpredictions = Symbol(Fprediction, "[$(smax+sstepsahead)]")
    cvlabel = haskey(Fθ2name, Fpredictions) ? Fθ2name[Fpredictions] : Symbol(:μ, "[$(smax+sstepsahead)]")
    cvlabels[Symbol(cvparamgroupname_pred_mean, "[$(smax-smaxstart+1)]")]= cvlabel
    cvlabels[Symbol(cvparamgroupname_err_mean, "[$(smax-smaxstart+1)]")]= cvlabel
    @assert typeof(dgpmeanbase) <: AbstractDGP{<:DGPMeanModelParameters}
    
    @unpack ŷ, se, newdgpbase = squarederror(Val{holdoutmethod}(); 
      dgpbase=dgpmeanbase, 
      dgpfull=dgpmean,
      smax, 
      Fprediction=:μ, 
      sstepsahead, 
      verbose, 
      numburnrecords,
      retrainingfrequency)

    #@eval Main ses_mean = $ses_mean

    @assert size(se) ≡ (numrecords, numchains)
    @assert size(ŷ) ≡ (numrecords, numchains)

    push!(ses_mean, reshape(se, numrecords, 1, numchains))
    push!(ŷs_mean, reshape(ŷ, numrecords, 1, numchains))
    dgpmeanbase = something(newdgpbase, dgpmeanbase)
  end


  addθnames!(dgp.dims; Fθ2name = cvlabels)
  #add the results to the dgp records. Adjust scaling for the expectation
  augmentations = Dict(
      cvparamgroupname_pred=>reduce(hcat,ŷs.* numsamplerecords*numchains),
      cvparamgroupname_pred_mean=>reduce(hcat,ŷs_mean .* numsamplerecords*numchains),
      cvparamgroupname_err=>reduce(hcat,ses.* numsamplerecords.*numchains),
      cvparamgroupname_err_mean=>reduce(hcat,ses_mean .* numsamplerecords*numchains),
      
      )

  augmented = augmentchainrecords(records, augmentations)
  #@eval Main augmented=$augmented
  #@eval Main augmentations=$augmentations
  #@eval Main records=$records
  #(holdoutmethod ≡ :truecv) && throw("examine")
  dgpnew = DGP(;Θ=dgp.Θ, data,  hyper, dims, records=augmented)


  cvte = sum((ses .|> sum).*cvweights)
  cvr2 = 1-sum((ses .|> sum).*cvweights)/sum((ses_mean .|> sum).*cvweights)

  @info "CV complete. Calculated cross-validated R2 of $cvr2"
  return (;cvr2, cvte, dgp=dgpnew)
end




function importanceratios(dgps::AbstractDGP; dgpbase, dgpfull, 
  Fldensity=PARAM[:cvFldensity],
  Fdensity=PARAM[:cvFdensity])::Matrix{Float64}

  @unpack numrecords, numchains=dgpfull.records
  #@eval Main dgps=$dgps
  ldensities::Matrix{Float64} = integrateposterior(dgps; records=dgpfull.records) do dgpi
    lpdist(dgpi, Fdensity) #note lpdist returns the log of the dsitribution
  end

  ldensitiesbase = if Fldensity ∈ dgpbase.records.fieldstocapture
    @assert Fldensity ∈ keys(dgpbase.records.chainparts)
    dgpbase.records[Fldensity]
  else
    @assert Fldensity ∉ keys(dgpbase.records.chainparts)
    integrateposterior(dgpbase; records=dgpfull.records) do dgpi
      lpdist(dgpi, Fdensity)
    end
  end

  #@eval Main ldensitiesbase = $ldensitiesbase

  lw̃ = reshape(ldensities, numrecords, numchains) - reshape(ldensitiesbase, numrecords, numchains)
  w = stableweights(;lw̃)

  if !isfinite(sum(w))
    @eval Main w=$w
    @eval Main lw̃=$lw̃
    @eval Main ldensities=$ldensities
    @eval Main ldensitiesbase=$ldensitiesbase
    @eval Main dgpbase=$dgpbase
    @eval Main dgps=$dgps
    throw("w̃ contains non-number")
  end
  #@eval Main push!(w,$w̃)=#

  @assert size(w) ≡ (numrecords, numchains)
  return w
end



#raw importance weights- likely unstable
function irweights(dgps::AbstractDGP, ::Val{:raw};  dgpbase, dgpfull, kwargs...)

  w = importanceratios(dgps; dgpbase, dgpfull)
  
  @assert sum(w) ≈ 1.0
  return (;w, converged=nothing, k=nothing)
end

  #ionides method- simple and parsimonious, but bias is higha ccording to Vehtari et al 2022
function irweights(dgps::AbstractDGP, ::Val{:ionides};  dgpbase, dgpfull )
  w̃ = importanceratios(dgps; dgpbase, dgpfull)
  Ew̃ = mean(w̃)
  q̃=w̃ .|> w̃i->min(w̃i, Ew̃ * sqrt(dgpbase.records.numsamplerecords*dgpbase.records.numchains))
  w = q̃ ./ sum(q̃)

  @assert sum(w) ≈ 1.0
  return (;w, converged=nothing, k=nothing)
end

loggpdpdf(x;μ,σ,k) = logpdf(GeneralizedPareto(μ,σ,k), x)

function fitgpd(x; 
  μ,
  k0=0.5,
  σ0=(mean(x)-μ)*(1-k0),
  loggpd = loggpdpdf,
  alg=Neldermead(), #seems to be fast and works well
  optkwargs...)

  function f(μ,σ,k) 
    (σ≤0.0) && return Inf
      -sum(loggpd(xi; μ,σ,k) for xi ∈ x)
  end

  f(args) = f(μ, args...)

  p0 = [σ0, k0,]
  opt = optimize(f, p0, alg, ;optkwargs...)

  σ,k = Optim.minimizer(opt)
  return(;μ,σ,k,flag=Optim.converged(opt))
end

#PSIS- follows Vehtari et al 2022
function irweights(dgps::AbstractDGP, ::Val{:psis};  dgpbase, dgpfull, testpsis=false)

  @unpack numsamplerecords=dgpbase.records

  w̃ = importanceratios(dgps; dgpbase, dgpfull)
  ṽ = testpsis ? vec(w̃) |> deepcopy : vec(w̃)


  #heuristic from Vehatri
  cutoff = round(min(numsamplerecords * 0.2,3*sqrt(numsamplerecords))) |> Int

  #identify the largest vals
  sortedinds = sortperm(ṽ)
  largestvalinds = sortedinds[(end-cutoff+1):end]

  #the μ parameter is the largest non-selected value
  μ = ṽ[sortedinds[end-cutoff]]
  largestvals = ṽ[largestvalinds]
  largest = maximum(largestvals)
  @assert largest == largestvals[end]
  @assert all(largestvals .≥ μ) 
  @assert all(μ .≥ ṽ[Not(largestvalinds)])

  #now estimate the GPD parameters
  #@unpack k, μ, σ, flag = fitgpd(x; alg=LBFGS(), autodiff=:forward)
  try
    @unpack k, σ, flag = fitgpd(largestvals; alg=NelderMead(), μ)
    if !flag
      @unpack k, σ, flag = fitgpd(largestvals; alg=BFGS(), μ, autodiff = :forward)
    end
    !flag && throw("convergence failed")

      #invert the cdf to to identify the smoothed values
    largestsmoothed = 1:cutoff .|> z->min(quantile(GeneralizedPareto(μ,σ,k), (z-0.5)/cutoff), largest)
    @assert all(μ .≤ largestsmoothed .≤ largest)
    ṽ[largestvalinds] .= largestsmoothed
    w = reshape(ṽ ./ sum(ṽ), size(w̃)...)
  @assert sum(w) ≈ 1.0
    if testpsis
      @eval Main w=$w
      @eval Main wraw=$(w̃ ./ sum(w̃))
      @eval Main (μ,σ,k) = $μ, $σ,$k
      @eval Main largestsmoothed=$largestsmoothed
      @eval Main largestvals=$largestvals
      @eval Main smax=$(dgps.dims.S)
    end
    @assert sum(w) ≈ 1.0
    return (;w, k, converged=true)
  catch convergenceerr
    @warn "Failed to converge on gpd parameters for chainid=$(
        dgpfull.records.chainid), smax=$(dgps.dims.S) with error $(convergenceerr). Falling back on ionides"
    @unpack w = irweights(dgps, Val{:ionides}(); dgpbase, dgpfull)
    return (;w, k=Inf, converged=false)
  end



  #throw("stop")


  @assert false
end


squarederror(::Val{:raw}; kwargs...)=squarederror(Val{:ir}(); smoothingmethod=:raw, kwargs...)
squarederror(::Val{:ionides}; kwargs...)=squarederror(Val{:ir}(); smoothingmethod=:ionides, kwargs...)
squarederror(::Val{:psis}; kwargs...)=squarederror(Val{:ir}(); smoothingmethod=:psis, kwargs...)

function squarederror(::Val{:ir}; smoothingmethod, dgpfull, dgpbase, smax, Fprediction, sstepsahead, kwargs...) 
  dgps = truncate(dgpfull; smax)
  @unpack dims= dgps
  @unpack S=dims
  @unpack y=dgpfull.data
  @unpack numrecords, numchains = dgpfull.records

  #sanity check on dimensions
  @assert smax ≡ dims.S
  @assert S ≤ dgpfull.dims.S - sstepsahead
  @assert dgpfull.dims.S == dgpbase.dims.S

  @unpack w, converged = irweights(dgps, Val{smoothingmethod}(); dgpbase, dgpfull)
  predictions = reshape(dgpbase.records[Fprediction],numrecords, numchains)

  if size(w) !== size(predictions)
    @eval Main predictions=$predictions
    @eval Main w=$w
    @eval Main dgpbase=$dgpbase
    throw("size of weights inconsistent with size of records prediction")
  end
  @assert sum(w) ≈ 1.0
  return (;ŷ=w.*predictions, 
    se=w.*((y[smax+sstepsahead] .- predictions).^2), 
    converged,
    newdgpbase=nothing)
end

function squarederror(::Val{:forwardadaptivepsis}; dgpfull, dgpbase, smax, Fprediction, sstepsahead, maxk=0.7, 
  numburnrecords, verbose, kwargs...)
  @unpack y=dgpfull.data
  @unpack numrecords,numchains=dgpfull.records
  
  if (dgpbase.dims.S < smax)
    @eval Main fieldstocapture = $dgpbase.records.fieldstocapture
    @assert Fprediction ∈ [dgpbase.records.fieldstocapture; expandchainfields(dgpbase.records, dgpbase.records.fieldstocapture)]

    @unpack k, w, converged = irweights(truncate(dgpfull; smax), Val{:psis}(); dgpbase, dgpfull)
    verbose && @info "Ran IR weights with smax=$smax, dgpbase.dims.S=$(dgpbase.dims.S). Resulting k=$k"
  end

  #if the psis calculation is unreliable or we haven't yet calculated a base backward in time
  if (dgpbase.dims.S ≥ smax) || (!something(converged, true)) || k > maxk
    verbose && @info "Recalculating base because $(
        dgpbase.dims.S ≥ smax ? "smax=$smax while dgpbase.dims.S=$(dgpbase.dims.S)" : "k ($k) < maxk ($maxk)")"
    dgpbase = predicttruncated(dgpfull; smax, Fprediction, sstepsahead, verbose, numburnrecords,  
      kwargs...)
    
    w = fill(1/(numrecords*numchains),numrecords, numchains)
  end

  predictions = reshape(dgpbase.records[Fprediction], numrecords, numchains)

  @assert sum(w) ≈ 1.0
  #return both the te and the new dgpbase
  return (;ŷ=w.*predictions, 
    se=w.*((y[smax+sstepsahead] .- predictions).^2), 
    newdgpbase=dgpbase)

end

squarederror(::Val{:forwardhybridraw}; kwargs...)=squarederror(
    Val{:forwardhybridir}(); smoothingmethod=:raw, kwargs...)
squarederror(::Val{:forwardhybridionides}; kwargs...)=squarederror(
    Val{:forwardhybridir}(); smoothingmethod=:ionides, kwargs...)
squarederror(::Val{:forwardhybridpsis}; kwargs...)=squarederror(
    Val{:forwardhybridir}(); smoothingmethod=:psis, kwargs...)

function squarederror(::Val{:forwardhybridir}; smoothingmethod, dgpfull, dgpbase, smax, Fprediction, sstepsahead, 
  retrainingfrequency, numburnrecords, verbose, kwargs...)
  @unpack y=dgpfull.data
  @unpack numrecords, numchains=dgpfull.records
  
  validir=false
  if ((dgpbase.dims.S < smax) && (smax-dgpbase.dims.S < retrainingfrequency))
    @eval Main fieldstocapture = $dgpbase.records.fieldstocapture
    @assert Fprediction ∈ [dgpbase.records.fieldstocapture; expandchainfields(dgpbase.records, dgpbase.records.fieldstocapture)]

    @unpack w, converged = irweights(truncate(dgpfull; smax), Val{smoothingmethod}(); dgpbase, dgpfull)
    verbose && @info "Ran IR weights with smax=$smax, dgpbase.dims.S=$(dgpbase.dims.S), converged=$(something(converged, "na"))"
    validir = something(converged,true)
  end
  if !validir
    verbose && @info "Recalculating base:  
      smax=$smax while dgpbase.dims.S=$(dgpbase.dims.S), retrainingfrequency=$retrainingfrequency"
    dgpbase = predicttruncated(dgpfull; smax, Fprediction, sstepsahead, verbose, numburnrecords,  
      kwargs...)
    
    w = fill(1/(numrecords*numchains), numrecords, numchains)
  end

  predictions = reshape(dgpbase.records[Fprediction], numrecords, numchains)
  @assert sum(w) ≈ 1.0
  #return both the te and the new dgpbase
  return (;ŷ=w.*predictions, 
    se=w.*((y[smax+sstepsahead] .- predictions).^2), 
    newdgpbase=dgpbase)

end

#run an mcmc for the training data
function predicttruncated(dgpfull;smax, Fprediction, sstepsahead, verbose, numburnrecords, 
  mcmcruntype=PARAM[:mcmcruntype], 
  kwargs...)

  dgps = truncate(dgpfull, :newrecords; smax, numburnrecords) 

    @unpack numsamplerecords, numrecords, numchains = dgpfull.records

  @unpack dims, data = dgps
  @unpack S = dims
  #@unpack s2t = dgpfull.dims

  spred = S+sstepsahead

  verbose && @info "Running true CV- predicting ($(smax+ sstepsahead)/$(dgpfull.dims.S))"

  dgps = mcmc(mcmcruntype, dgps; 
    stoprule=:bootstrapess, 
    maxrecords=numburnrecords+numsamplerecords, 
    verbose=false)

    #we can burn the records before computing derived stats, since we won't report convergence stats
  dgps = m.DGP(dgps, records=burnchainrecords(;dgp=dgps ), strict=false)



  #compute the prediction if needed
  if Fprediction ∉ [propertynames(dgps.Θ); expandchainfields(dgps.records, propertynames(dgps.Θ) |> collect)]
    Fpredictiongroup = replace("$Fprediction",r"\[[0-9]+\]"=>"") |> Symbol
    #=preddata = FrData(;
      F=dgpfull.data.F,
      r=dgpfull.data.r,)=#
    dgps=addderived(Fpredictiongroup;dgp=dgps,  data=dgpfull.data, dims=dgpfull.dims)
  end
  
  #if the prediction field is present already, no need to remake it
  #=if Fprediction ∉ [propertynames(dgps.Θ); expandchainfields(dgps.records, propertynames(dgps.Θ) |> collect)]
    dgps = appendprediction(dgps; dgpfull, Fprediction, spred)
  end=#

  
  return dgps
end

#=function appendprediction(dgps; dgpfull, Fprediction, spred)
  @unpack s2t = dgpfull.dims
  @unpack dims, data = dgps

  Fpredictiongroup = replace("$Fprediction",r"\[[0-9]+\]"=>"") |> Symbol
  preddata = FrData(;
    F=dgpfull.data.F,
    r=dgpfull.data.r,)
  preddims = Dims(dims;S=spred, dates=nothing, Fθ2name=nothing,addxylabels=nothing)
  dgps=addderived(Fpredictiongroup;dgp=dgps,  data=preddata, dims=preddims)

  return dgps
end=#

function squarederror(::Val{:truecv}, ; smax, Fprediction,
  sstepsahead,
  dgpfull,
  kwargs...)

  @unpack y=dgpfull.data
  @unpack numrecords, numchains=dgpfull.records

  dgps = predicttruncated(dgpfull; smax, Fprediction, sstepsahead, kwargs...)

  @assert dgps.dims.S == smax

  #weights are uniform in this scenario
  w = fill(1/(numrecords*numchains), numrecords, numchains)

  
  predictions = reshape(dgpbase.records[Fprediction], numrecords, numchains)
  @assert sum(w) ≈ 1.0

  return (;ŷ=w.*predictions, 
    se=w.*((y[smax+sstepsahead] .- predictions).^2), 
    newdgpbase=nothing)
end



testcv() = simulatechainanalysis(;) |> s->testcv(s.dgp; rid=s.rid, outputpath=s.outputpath)

function testcv(dgp::AbstractDGP; rid, outputpath)

  analyzemcmcoutput(;unburnt=dgp, rid, outputpath, cvverbose=true)

end

