




#runs a simple block bootstrap for a set of statistics
#this and the helper function form the bootstrap core
function simpleblockbootstrap(v::AbstractVector, 
    statsfordraw::Tstatsfordraw, ::Type{Tresult}=Float64; 
    blocksize=PARAM[:bootstrapblocksize], 
    numdraws=PARAM[:bootstrapnumdraws],
    results::Vector{Tresult} = Vector{Tresult}(undef, numdraws),
    ) where {Tstatsfordraw <: Function, Tresult}

  T = length(v)


  #draw a sample and process the results
  Threads.@threads for i ∈ 1:numdraws
    vi = v[drawindices(;blocksize, T)]
    results[i] = statsfordraw(vi)
  end

  return results
end

#helper function to draw the index runs
function drawindices(; blocksize, T)  
  tinds::Vector{Int} =  sample(1:(T-blocksize+1), T, replace=true)
  for t ∈ 1:T
    ((t-1) % blocksize == 0) && continue
    tinds[t] = tinds[t-1]+1
  end

  return tinds
end


#these are estimatoes for the correlation- they average over the different chains
#if different chains are provided
corfunc_covvar(θ::AbstractVector{<:Union{Missing, Real}}; l, σ2) = cov(θ[1:(end-l)], θ[(l+1):end])/σ2
corfunc_covvar(θ::AbstractVector{<:AbstractVector}; kwargs...) = mean(θ .|> θi->corfunc_covvar(θi; kwargs...))

function corfunc_variogram(θ::AbstractVector{<:Union{Missing, Real}};l, σ2, T=length(θ))
  
  θt = @view θ[(l+1):end]
  θtδ = @view θ[1:(end-l)]
  return 1.0-mean(abs2.(θt .- θtδ))/(2*σ2) 
end
corfunc_variogram(θ::AbstractVector{<:AbstractVector}; kwargs...) = mean(θ .|> θi->corfunc_variogram(θi; kwargs...))


#a simple heuristic for identifying a low correlation lag
function findlowcorlag(θ; minlag=0, maxlag, acceptableautocor, 
  testfindlowcorlag=false, 
  lagmultiplier = 1.0,
  σ2=var(θ),
  corestimator = :covvar,
  forcemonotoneautocors=false,
  estimatorkwargs...)


  T = length(θ)
  corfunc(l) = Dict(
      :covvar=>corfunc_covvar, 
      :variogram=>corfunc_variogram,
      )[corestimator](θ; l, σ2, estimatorkwargs...)

  local lag::Int

  #local lagautocors = recordautocors ? [nextlagautocor] : Float64[]
  local lagautocors::Vector{Float64} = Float64[]
  sizehint!(lagautocors, maxlag)
  push!(lagautocors, corfunc(1)) #initialize
  local cumlagautocor = lagautocors[end]
  local marginallagautocor = Inf

  local acceptablelagfound::Bool = false

  #the heuristic is such that we seek the first lag s.t. the average of the next two autocorrelations are negative
  lags = 1:2:maxlag
  for b ∈ lags
    lag=b
    
    #estimate the autocorrelation using the chosen estimator
    lagautocorP1 = corfunc(lag+1)
    lagautocorP2 = corfunc(lag+2)

    #if the subsequent autocors average to less than the acceptable autocor, we are done
    #note- only even lags can be selected
    if (lag≥minlag) && ((lagautocorP1+lagautocorP2)/2 < acceptableautocor)
      acceptablelagfound=true
      break
    end

    marginallagautocor = ifelse(forcemonotoneautocors, 
      min(marginallagautocor, lagautocorP1+lagautocorP2), 
      lagautocorP1+lagautocorP2)

    cumlagautocor += marginallagautocor
    
    append!(lagautocors, [lagautocorP1, lagautocorP2])
  end

  lagautocor = lagautocors[end]


  #a simple testing routine
  if length(lags)==1
    @assert lag = minlag
  elseif testfindlowcorlag
    if corestimator==:covvar
      autocors = (1:(maxlag+2) .|> b->cov(θ[1:(end-b)], θ[(b+1):end],)) ./ var(θ)
    elseif corestimator==:variogram
      n = minimum(length.(θ))
      autocors = mean(θ .|> (θi-> 1 .- 
        @views (1:(maxlag+2) .|> (b->
          sum((θi[1:(end-b)] .- θi[(b+1):end]).^2) ./  (σ2*(n-b)*2)))
        ))
    end

    avgnext2s = 1:2:maxlag .|> b->(autocors[b+1]+autocors[b+2])/2
    if forcemonotoneautocors
      avgnext2s .= accumulate(min, avgnext2s)
    end
    acceptablelags = collect(1:2:maxlag)[(avgnext2s .< acceptableautocor) .& (1:2:maxlag .|> b-> b≥minlag)]

    #@eval Main lags = $lags
    #@eval Main autocors = $autocors
    #@eval Main lagautocor = $lagautocor
    lagend = lags[end]

    @assert lag==(!isempty(acceptablelags) ? acceptablelags[1] : maximum(lags)) "
      lag=$lag but acceptablelags[1]=$(acceptablelags[1])"
    lagptr = (findfirst(isequal(lag), lags)-1)*2+1
    #@eval Main lagptr=$lagptr

    @assert ((autocors[lagptr] ≈ lagautocor) || (autocors[lagptr] ≡ lagautocor ) ||
      (lagptr==lagend && autocors[lagptr+2] == lagautocor)) "
      autocors[.]=$(autocors[lagptr]) but lagautocor=$lagautocor"

    cumlagautocorcheck = (forcemonotoneautocors ? 
      [autocors[1];autocors[1] .+ cumsum(avgnext2s) .* 2][findfirst(isequal(lag), lags)]  :
      sum(autocors[1:lagptr]))

    #might fail due to monotonicity if the endpoint is reached? If so, debug the check but its not a bug in the actual code
    @assert cumlagautocor ≈ cumlagautocorcheck || (cumlagautocor ≡ cumlagautocorcheck) || (
      lagptr==lagend && autocors[lagptr+2] + autocors[lagptr+1] + cumlagautocorcheck ≈ cumlagautocor) " 
      cumlagautocor=$cumlagautocor  but cumlagautocorcheck = $cumlagautocorcheck !!"
  end

  #using a different lag size is helpful in some cases, particularly if an ma process is suspected
  lag=min(max(1, lag*lagmultiplier |> Int), maxlag)
  lagautocor = corfunc(lag)
  acceptablelagfound |= lagautocor<acceptableautocor
  
  return (;lag, lagautocor, cumlagautocor, lagautocors, acceptablelagfound)
end

#NOTE- this takes a matrix-form parameter (iterations x chains) and formstats
#an array of vectors, one for each chain, and optionally splits the chains in two
function formchains(θ::AbstractMatrix{Tθ}; splitchains::Bool) where Tθ<:Union{Real,Missing}
  T = size(θ,1)
  numchains = size(θ,2)
  θtot = vec(θ)

  #form the split chains
  if splitchains
    p = T ÷ 2
    θs = [θ[1:p,:] |> eachcol |> collect; θ[(p+1):(2*p),:] |> eachcol |> collect]
    chainnames = [1:numchains .|> i->Symbol(:chn, i, :a); 1:numchains .|> i->Symbol(:chn, i, :b)]

  else
    θs = θ |> eachcol |> collect
    chainnames = [1:numchains .|> i->Symbol(:chn, i); ]
  end

  @assert allequal(θs .|> length)

  return (;θtot, θs, chainnames, )
end

function formconditionalchains(θ::AbstractMatrix{Tθ}, θkeep) where Tθ<:Union{Real,Missing}
  numchains = size(θ,2)
  @assert size(θ) ≡ size(θkeep)
  θtot = vec(θ[θkeep])

  #form the conditional chains
  θs = map(θ |> eachcol, θkeep |> eachcol) do θc, θckeep
    return θc[θckeep]
  end
    
  chainnames = [1:numchains .|> i->Symbol(:chn, i); ]

  #@assert allequal(θs .|> length)

  return (;θtot, θs, chainnames, )
end

#NOTE: this replicates the BDA stats from the MCMC diagoniistic package-
#The only difference is the below by default does not force a monotone sequence
#unless replicate package is set to true
#POSSIBLE Enhancement- see https://arxiv.org/pdf/1903.08008.pdf - implement the rank normalziation
#scheme from the package (setting kind=:bulk)
function bdastats(θs::AbstractVector{Tθ}; 
  testbdastats=PARAM[:testbdastats], 
  replicatemcmcdiagnosticsbda = PARAM[:diagreplicatemcmcdiagnosticsbda],
  maxbdalag = PARAM[:diagmaxbdalag],
  kwargs...) where Tθ<:AbstractVector

  #follow the notation of BDA
  m = length(θs)
  ns = length.(θs)
  n = minimum(ns)
  @assert maximum(abs.(ns .- n)) ≤ 1

  all(n .== ns) || @warn("size of chains is not equal- results amy be inconsistent with chain lengths $ns")
  #θs = deepcopy(θs) .|> θj->θj[1:minimum(ns)]

  μs = θs .|> mean
  μ = μs |> mean
  B = n * var(μs, corrected=true)
  s2 = ((θj, μj)-> var(θj, mean=μj, corrected=true)).(θs,μs)
  W = s2 |> mean

  #next compute the variance using the BDA method (11.3, pg284)
  σ2 = (n-1)/n*W+1/n*B
  Rscale = sqrt(σ2/W)

  @unpack cumlagautocor, lagautocor, lag, lagautocors =  findlowcorlag(θs .|> θ->θ .- mean(θ); 
    maxlag=min(n - 2,maxbdalag), 
    acceptableautocor=0.0,
    corestimator=:variogram,
    σ2,
    forcemonotoneautocors=replicatemcmcdiagnosticsbda,
    testfindlowcorlag = testbdastats,
    kwargs...)
  ess = m*n/(1+2*cumlagautocor)
  se = ((1+2*cumlagautocor)/(m*n) * σ2)^0.5

  if testbdastats
    @assert B ≈ n/(m-1) * sum(abs2.(μs .- μ))

    Wrep=1/m * 1/(n-1)* sum(((θi,μi)->sum(abs2.(θi .- μi))).(θs, μs))
    @assert W ≈ Wrep "W=$W but Wrep=$Wrep !!"

    cumlagautocors = cumsum(lagautocors)
    esss = m*n ./ (1 .+ 2 .* cumlagautocors)
  else
    esss=Float64[]
  end


  return (; σ2, cumlagautocor, lagautocor, lag, ess, se, Rscale, esss)
end




#compute the summary statistics for each param
function formstats(θ::AbstractVector; Fθ, Fchain, 
  simplestat,
  summaryconvergencestats=true, 
  minblocklag=PARAM[:bootstrapminblocklag],
  maxblocklag=PARAM[:bootstrapmaxblocklag],
  acceptableautocor=PARAM[:bootstrapacceptableautocor],
  quantiles=PARAM[:mcmcquantiles],)

  chainlength=length(θ)
  θstats = OrderedDict()
  θstats[:param] = Fθ
  θstats[:chain] = Fchain

  if chainlength == 0
    θstats[:N] = length(chainlength)
    return θstats
  end
  

  θstats[:E] = mean(θ)
  θstats[:med] = median(θ)
  
  θstats[:min] = minimum(θ)


  for q ∈ quantiles
    leftofpoint = q*100 |> floor |> Int
    rightofpoint = (q*1_000 |> floor |> Int)-leftofpoint*10
    #try
      θstats[Symbol(:q, leftofpoint, rightofpoint==0 ? "" : "p$rightofpoint")] = quantile(θ,q)
    #=catch err
      @eval Main θ=$θ
      @warn("quantile calc failed for stat $Fθ with error $err")
    end=#
  end


  θstats[:max] = maximum(θ)
  
  θstats[:sd] = std(θ)
  θstats[:naivese] = θstats[:sd]/chainlength^0.5 #assumes iid

  if summaryconvergencestats && (!simplestat)

    #this correlation serves to determine the block size for the ess calc (and other bootstrap stats if desired later)
    blocksize, blockautocor = findlowcorlag(θ; 
      minlag=minblocklag, 
      maxlag=maxblocklag, 
      acceptableautocor, 
      lagmultiplier=PARAM[:bootstraplagmultiplier]) |> nt->(nt.lag+1, nt.lagautocor)

    θstats[:blocksize] = blocksize
    θstats[:blockautocor] = blockautocor

    #probably the best way to compute the SE
    θstats[:bbs_se] = std(simpleblockbootstrap(θ,mean; blocksize))
    θstats[:bbs_ess] = (θstats[:sd]/θstats[:bbs_se])^2
  end

  θstats[:N] = chainlength

  #θstats[:Elog] = mean(θ .|> log)
  #θstats[:varlog] = var(θ .|> log)

  return θstats
end

function formstats(;θmat, θtot, θs, chainnames, Fθ, simplestat, splitchains, summaryconvergencestats, )

  numchains = length(θs)

  #slow but shouldn't matter much (though could be parallelized...)
  θsstats = ((θ, Fchain)->
    formstats(θ; Fθ, Fchain, summaryconvergencestats, simplestat)).(θs, chainnames
    ) |> ds->collapse(ds;strict=false) |> DataFrame

  #@eval Main θsstats=$θsstats
  #now the grouped diagonistics
  θstats = formstats(θtot; Fθ, Fchain=:full, simplestat, summaryconvergencestats=false)

  simplestat && return vcat(θsstats, DataFrame(θstats), cols=:union)

  if summaryconvergencestats
    if numchains == 1
      θstats[:blocksize] = θsstats.blocksize[1]
      θstats[:blockautocor] = θsstats.blockautocor[1]
    end
    θstats[:bbs_se] = (mean(θsstats.bbs_se .^2) / numchains) ^0.5
    θstats[:bbs_ess] = (θstats[:sd]/θstats[:bbs_se])^2
    θstats[:bbs_esstot] = sum(θsstats.bbs_ess)

    #other essstats
    #using the package requires some reshaping

  end
  

  θmat3d = reshape(θmat,size(θmat,1),size(θmat,2),1)
  θstats[:geyer_ess], θstats[:geyer_Rscale] = MCMCChains.MCMCDiagnosticTools.ess_rhat(θmat3d, 
    autocov_method=MCMCChains.MCMCDiagnosticTools.AutocovMethod(), 
      kind=PARAM[:diagpackagebdakind]#=kind=:basic=#)|> t->(t[1][1],t[2][1])
  
  θstats[:bda_ess], θstats[:bda_Rscale] = MCMCChains.MCMCDiagnosticTools.ess_rhat(θmat3d, 
    autocov_method=MCMCChains.MCMCDiagnosticTools.BDAAutocovMethod(), 
    kind=PARAM[:diagpackagebdakind]#=kind=:basic=#)|> t->(t[1][1],t[2][1])

  #this methedology requires multiple chains
  if (length(θs) > 1 ) && summaryconvergencestats
    bda2 = bdastats(θs;)

    #should always hold when the chains are equal length, otherwise the check creates issues
    #@assert (θstats[:bda_Rscale] ≈ bda2.Rscale) || (θstats[:bda_Rscale] ≡ bda2.Rscale) || !splitchains "
    #  θstats[:bda_Rscale]=$(θstats[:bda_Rscale]) but bda2.Rscale=$(bda2.Rscale)!!"

    θstats[:bda2_ess] = bda2.ess
    θstats[:bda2_ess_sd] = bda2.σ2^0.5
    θstats[:bda2_ess_se] = bda2.se
    θstats[:bda2_lag] = bda2.lag
    θstats[:bda2_lagautocor] = bda2.lagautocor

    #this is memory intensive and only for debugging
    if length(bda2.esss) > 0
      θstats[:bda2_esss] = [bda2.esss[1:min(length(bda2.esss),500)]]
    end

  end

  θsstats = vcat(θsstats, DataFrame(θstats), cols=:union)

  return θsstats
end

function summarizechainoutput(;
  dgp,
  fieldstoanalyze=dgp.records.fieldstocapture,
  splitchains::Bool,
  slocalparams=PARAM[:ioslocalparams],
  slocalparams_full=PARAM[:ioslocalparams_full],
  tlocalparams=PARAM[:iotlocalparams],
  tlocalparams_full=PARAM[:iotlocalparams_full],
  simplelocalstatsoverride=PARAM[:iosimplelocalstatsoverride], 
  additionalsimplestats=PARAM[:ioadditionalsimplestats], 
  simplelocalstats=PARAM[:iosimplelocalstats],
  summaryconvergencestats=PARAM[:iosummaryconvergencestats],
  conditionalparams=PARAM[:ioconditionalparams])

  @unpack hyper, records, dims, data = dgp

  @unpack numchains,numsamplerecords, chain, lastrecord,chainfieldindex = records
  @unpack T,S,Fθ2name,K, priorset = dims
  @unpack y = data

  simplestats = vcat(
    simplelocalstats ? [slocalparams; slocalparams_full; tlocalparams; tlocalparams_full] : Symbol[],
    additionalsimplestats,
  ) |> ss-> setdiff(ss, simplelocalstatsoverride)



  local stats::DataFrame = DataFrame()


  #groupedfieldnames = Dict(f=>namesingroup(chain, f) for f ∈ fieldstocapture)
  #rdf = chain |> DataFrame
  chainlength = size(chain,1)
  @assert chainlength == numsamplerecords "numsamplerecords != rows in record- are the records unburnt?"



  for fgroup ∈ fieldstoanalyze
    for Fθ ∈ chainfieldindex[fgroup]
      θmat = chain[:,Fθ,:] |> Matrix{Float64}

      ##main entry point for computing the summary stats
      vchains = formchains(θmat; splitchains, )

      simplestat = (Fθ ∈ simplestats) || (fgroup ∈ simplestats)
      #**main call to compute stats
      additionalstats = formstats(;θmat, Fθ, splitchains, summaryconvergencestats, 
        simplestat, vchains...)
      additionalstats.paramgroup .= fgroup

      #grab any conditional stats we desire
      if :γ ∈ fieldstoanalyze && fgroup ∈ conditionalparams
        groupindex = match(r"\[[0-9]+\]", "$Fθ").match
        Fγ = Symbol(:γ, groupindex)
        γmat = chain[:,Fγ,:] |> Matrix{Bool}
        vconditionalchains = formconditionalchains(θmat, γmat)
        Fθc = Symbol(:cγ_, Fθ)
        conditionalstats = formstats(;
          θmat=nothing, 
          Fθ=Fθc, 
          splitchains=false, 
          summaryconvergencestats=false, 
          simplestat=true, 
          vconditionalchains...)  
        conditionalstats.paramgroup .= Symbol(:cγ_, fgroup)
        additionalstats = vcat(additionalstats, conditionalstats, cols=:union)
      end   
        

      #cleans up the single chain case (hacky)
      if (!splitchains) && (numchains==1)
        additionalstats=additionalstats[additionalstats.chain .!== :chn1, :]
        @assert all(additionalstats.chain .≡ :full)
        additionalstats.chain .= :chn1
      end

      stats = vcat(stats, additionalstats, cols=:union)


    end
  end

    
  #append notes
  #@eval Main f=$Fθ2name
  #throw("stop")
  if !isempty(Fθ2name)
    statsnotes = DataFrame(param=collect(keys(Fθ2name)))
    statsnotes.note = statsnotes.param .|> Fθ->Fθ2name[Fθ]
    stats = leftjoin(stats, statsnotes, on=:param)
  end


  stats.v .= hyper.v
  stats.priorset .= priorset
  predictors = intersect(stats.paramgroup |> unique, [slocalparams; slocalparams_full])

  #this will prove helpful when aggregatig
  if !isempty(predictors) && haskey(Fθ2name, Symbol("y[1]"))
    #stats.y = missings(Float64, nrow(stats))
    ydf = crossjoin(DataFrame(y=y, note=[Fθ2name[Symbol("y[$s]")] for s ∈ 1:S],), DataFrame(paramgroup=predictors))
    #@eval Main ydf=$ydf
    stats = leftjoin(stats, ydf, on=[:note, :paramgroup], matchmissing=:notequal)
    #@eval Main stats=$stats
    for predictor ∈ predictors
      @assert all(stats[stats.paramgroup .≡ predictor, :note] .!== missing)
    end
    #=for predictor ∈ predictors, chn ∈ unique(stats.chain)
      predictorstats = @view(stats[(stats.paramgroup .== predictor) .& (stats.chain .== chn),:])
      @assert issorted(predictorstats, :pid)
      predictorstats.y .= y
    end=#
  end



  extractparamprior(p::AbstractVector,ind::Int)=p[ind]
  extractparamprior(p::AbstractMatrix,ind::Int)=p[ind,ind] #get the diagonals if a matrix
  extractparamprior(p, args...) = p

  #attatch the priors to the dataframe for reference
  priorstats = DataFrame(param=Symbol[], prior=NamedTuple[])
  hypernames = fieldnames(typeof(hyper)) 
  for f ∈ hypernames
    f ∉ fieldstoanalyze && continue
    hyperf = hyper[f]
    isempty(hyperf) && continue

    fexpanded = expandchainfield(records, f)
    #iterate over all of the parameters associated with the field (eg β has K parameters, τy has 1)
    for i ∈ 1:length(fexpanded)
      paramprior=Dict()
      for Fp ∈ propertynames(hyperf)
        prior = extractparamprior(hyperf[Fp],i)
        paramprior[Fp] = prior
      end
      push!(priorstats, (;param=length(fexpanded) == 1 ? f : Symbol(f,"[$i]"), prior = (;paramprior...)))
    end
  end
  stats = leftjoin(stats, priorstats, on=:param)



  #=if :β ∈ fieldstoanalyze
    βpriorstats = DataFrame(param = (1:K .|> k->Symbol("β[$k]")), β0=hyper.β0 .|> Float64, βΔ0=hyper.βΔ0 .|> Float64)
    stats = leftjoin(stats, βpriorstats, on=:param)
  end=#

      
  #reorder the columns and sort
  flattenedfields = reduce(vcat, fieldstoanalyze .|> fgroup->chainfieldindex[fgroup])
  @assert nrow(stats) == (length(flattenedfields)+K*length(intersect(fieldstoanalyze, conditionalparams)))*length(unique(stats.chain))
  stats=leftjoin(stats, DataFrame(pid=1:length(flattenedfields), param=flattenedfields), on=:param)
  sort!(stats, [:pid, :chain])

  select!(stats, [:paramgroup; :param; :note; setdiff(propertynames(stats), [:paramgroup, :note,:param, :pid])])    
  return stats
end




function createaggregatesummaryfile(;
  Fpredictivestats=PARAM[:iopredictivestats],
  Faggregatestats=PARAM[:ioaggregatestats],
  rid=PARAM[:batchiorid],
  analysispath=PARAM[:analysispath],
  predictionmethods=PARAM[:iopredictionmethods],
  computedensity=PARAM[:iocomputedensity],
  cvoverrides = PARAM[:iocvoverrides],)




  summaryfiles = readdir("$analysispath/$rid/summary/whole")
  filter!(f->match(Regex("s_[a-zA-Z0-9_]*$rid.csv"),f) !== nothing, summaryfiles)
  dateindex = Dict{Symbol, Date}()
  paramgroupindex = Dict{Symbol, Symbol}()

  agg = DataFrame()
  for f ∈ summaryfiles
    d = CSV.File("$analysispath/$rid/summary/whole/$f") |> DataFrame
    d.mean = d.E .|> x->x≡missing ? missing : x
    d.median = d.med  .|> x->x≡missing ? missing : x
    d.param = d.param .|> Symbol
    d.paramgroup = d.paramgroup .|> Symbol
    d.N = d.N .|> n->coalesce(Int(n),missing)
    d.prior = d.prior  .|> p->p ≢ missing ? string(p) : missing
    
    Fpredictions = intersect(unique(d.paramgroup), predictionmethods )

    #create a new row for each chain
    for chainname ∈ unique(d.chain)
      dchain = d[d.chain .== chainname,:]

      #these will be helpful for organizing the aggregates
      unorderednotes = Dict(dchain[!, :param] .=> dchain[!, :note])
      unorderedparamgroups = Dict(dchain[!, :param] .=> dchain[!, :paramgroup])
      unorderedE = Dict(dchain[!, :param] .=> dchain[!, :E])

      numsamples = maximum(skipmissing(dchain.N))
      priorset = dchain[begin, :priorset]
    
      stats=OrderedDict()
      stats[:fund] = replace(f, "s_A_"=>"", "_$rid.csv"=>"", )
      stats[:chain] = chainname
      stats[:N] = numsamples
      stats[:priorset] = priorset


      uniqueparams = unique(dchain.paramgroup)

      if :x ∈ uniqueparams
        stats[:T] = nrow(@views dchain[dchain.paramgroup .== :x, :])
      end

      if !isempty(Fpredictions)
        stats[:S] = nrow(@views dchain[dchain.paramgroup .== Fpredictions[begin], :])
      end

      if :ϕ ∈ uniqueparams
        stats[:P] = nrow(@views dchain[dchain.paramgroup .== :ϕ, :])
      end

      if :β ∈ uniqueparams
        stats[:K] = nrow(@views dchain[dchain.paramgroup .== :β, :])
      end 
      
      #this backs out Δt, and performs a check on the other dimensions
      if haskey(stats, :T) && haskey(stats, :S) && haskey(stats, :P)
        stats[:Δt] = (stats[:T]-stats[:P])/(stats[:S]) |> Int


        for cvoverride ∈ cvoverrides
          cvparams = loadcvparams(; S=stats[:S], Δt=stats[:Δt], P=stats[:P], cvoverride...)
          @unpack cvparamgroupname, cvparamgroupname_err, cvparamgroupname_err_mean, cvparamgroupname_pred, cvparamgroupname_pred_mean = cvparams
          @unpack cvweights, sstepsahead, smaxstart, smaxend, Stest = cvparams
          cvFprediction = cvparams.Fprediction
          @assert sum(cvweights) ≈ 1.0

          if cvparamgroupname_err ∈ uniqueparams

            Eses = dchain[dchain.paramgroup .== cvparamgroupname_err, :E]
            Eses_mean = dchain[dchain.paramgroup .== cvparamgroupname_err_mean, :E]

            #Bayesian R2 metric
            stats[Symbol(cvparamgroupname,:_te)] = sum(Eses.*cvweights)^0.5
            stats[Symbol(cvparamgroupname,:_r2)] = 1-sum(Eses.*cvweights)/sum(Eses_mean.*cvweights)



            if length(Fpredictions) ≥ 1 #(need to have captured one of these in order to have recorded a value of y)
              y=dchain[dchain.paramgroup .== Fpredictions[begin], :y] 
              #Bayesian CV R2 metrics
              ŷs = dchain[dchain.paramgroup .== cvparamgroupname_pred, :E]
              ŷs_mean = dchain[dchain.paramgroup .== cvparamgroupname_pred_mean, :E]
              @assert length(ŷs) == length(ŷs_mean) ==Stest == smaxend-smaxstart+1
              @assert smaxend+sstepsahead==stats[:S] 

              
              @unpack ŷs_freqse, ŷs_meanfreqse, ŷs_mean0modelse,ŷs_rss = map(smaxstart:smaxend) do smax
                ŷs_freqse = (ŷs[smax-smaxstart+1] - y[smax + sstepsahead])^2
                ŷs_meanfreqse = (ŷs_mean[smax-smaxstart+1] - y[smax + sstepsahead])^2
                ŷs_mean0modelse  = (mean(y[1:smax]) - y[smax + sstepsahead])^2
                ŷs_rss = (ŷs[smax-smaxstart+1] .- mean(y[1:smax])).^2

                return (;ŷs_freqse, ŷs_meanfreqse, ŷs_mean0modelse,ŷs_rss )
              end |> collapse

              #@eval Main ŷs_freqse=$ŷs_freqse

              #Naive cvr2 metrics
              stats[Symbol(cvparamgroupname,:_freqte)] = sum(ŷs_freqse.*cvweights).^0.5
              stats[Symbol(cvparamgroupname,:_freqr2)] = 1-sum(ŷs_freqse.*cvweights)/sum(ŷs_meanfreqse.*cvweights)
              stats[Symbol(cvparamgroupname,:_0modelr2)] = 1-sum(ŷs_freqse.*cvweights)/sum(ŷs_mean0modelse.*cvweights)

              #stats[Symbol(cvparamgroupname,:_brssr2)] = sum(ŷs_rss.*cvweights)/sum((ŷs_rss .+ Eses).*cvweights)



              isresr2_label = Symbol(:yF_S,Stest,:_isresr2)
              if cvFprediction ∈ uniqueparams && isresr2_label ∉ keys(stats)
                ŷsis = dchain[dchain.paramgroup .== cvFprediction, :E]
                @assert length(ŷsis) == length(y)
                stats[:isresr2_label] = (1 - sum(cvweights.*(y[smaxstart:smaxend] .- ŷsis[smaxstart:smaxend]).^2)
                  / sum(cvweights .* (y[smaxstart:smaxend] .- mean(y[smaxstart:smaxend])).^2))
              end
            end
            



          end


        end

      end

      #Bayes factor??
      #if isempty(setdiff(:postinv, :postinv_mean, :post, :post_mean))
        #stats[:bayesactvmean] = ??
      #end 

      #these our our main volatility metrics
      if :sum2x_calc ∈ uniqueparams && haskey(stats, :T) && haskey(stats, :S)

        #cross-secitonal volatility e.g. variance of the posterior
        stats[:x_Xcalcσ] =  sqrt(unorderedE[:sum2x_calc]/stats[:T] - (unorderedE[:sumx_calc]/stats[:T])^2)
        stats[:x_Xcalcσ_adj] =  sqrt(unorderedE[:sum2x_calc_adj]/stats[:T] - (unorderedE[:sumx_calc]/stats[:T])^2)
        stats[:y_Xcalcσ] =  sqrt(unorderedE[:sum2y_calc]/stats[:S] - (unorderedE[:sumy_calc]/stats[:S])^2)
        
        stats[:x_Xcalcσ_scaledsqrt12] = stats[:x_Xcalcσ]*12^0.5
        stats[:x_Xcalcσ_adj_scaledsqrt12] = stats[:x_Xcalcσ_adj]*12^0.5
        stats[:y_Xcalcσ_scaled2xsqrt12] = (stats[:y_Xcalcσ]/sqrt(stats[:Δt]))*12^0.5

        #these look at the expected vol- the vol investors should expect to experience
        stats[:x_calcσ_scaledsqrt12] = unorderedE[:σ2x_calc]^0.5*12^0.5
        stats[:x_calcσ_adj_scaledsqrt12] = unorderedE[:σ2x_calc_adj]^0.5*12^0.5
        stats[:y_calcσ_scaled2xsqrt12] = (unorderedE[:σ2y_calc]/stats[:Δt])^0.5*12^0.5

        #longer run variance
        stats[:σxε_calc_adj_scaledsqrt12] = unorderedE[:σ2xε_calc_adj]^0.5*12^0.5
        stats[:σlrsys_calc_scaledsqrt12] = unorderedE[:σ2lrsys]^0.5*12^0.5

      end

      #different version of mcuh of the above
      if :sumx2_calc2 ∈ uniqueparams && haskey(stats, :T) && haskey(stats, :S)
        stats[:x_calc2σ] =  sqrt(unorderedE[:sum2x_calc2]/stats[:T] - (unorderedE[:sumx_calc2]/stats[:T])^2)
        stats[:x_calc2σ_adj] =  sqrt(unorderedE[:sum2x_calc2_adj]/stats[:T] - (unorderedE[:sumx_calc2]/stats[:T])^2)
        stats[:y_calc2σ] =  sqrt(unorderedE[:sum2y_calc2]/stats[:S] - (unorderedE[:sumy_calc2]/stats[:S])^2)
        
        stats[:x_calc2σ_scaledsqrt12] = stats[:x_calc2σ]*12^0.5
        stats[:x_calc2σ_adj_scaledsqrt12] = stats[:x_calc2σ_adj]*12^0.5
        stats[:y_calc2σ_scaled2x] = stats[:y_calc2σ]/sqrt(stats[:Δt])
        stats[:y_calc2σ_scaled2xsqrt12] = stats[:y_calc2σ_scaled2x]*12^0.5
      end

      if :yF ∈ Fpredictions && computedensity
        pyF = dchain[dchain.paramgroup .== :pyF, :E]
        stats[:yF_lppd] = (pyF .|> log) |> sum
      end


      #predictors are split out into columns
      for Fy ∈ Fpredictions
        dpred = dchain[dchain.paramgroup .== Fy, :]
        y=dpred.y        
        @assert length(y) ≡ nrow(dpred)

        if !haskey(stats, :σy_hist)
          stats[:σy_hist] = std(y)
          stats[:σy_hist_scaled2x] = stats[:σy_hist]/sqrt(stats[:Δt])
          stats[:σy_hist_scaled2xsqrt12] = stats[:σy_hist_scaled2x]*12^0.5
        end

        
        for Fstat ∈ Fpredictivestats
          yhat = dpred[!,Fstat]

          stats[Symbol(Fstat, Fy, :_r2)] = cor(yhat,y)^2
          tss = sum((y .- mean(y)).^2)
          rss = sum((yhat .- mean(y)).^2)
          stats[Symbol(Fstat, Fy, :_r2_rss)] = rss/tss
          stats[Symbol(Fstat, Fy, :_r2_res)] = 1-sum((y .- yhat).^2)/tss
          stats[Symbol(Fstat, Fy,  :_te)] = (yhat .- y).^2 |> mean |> sqrt


          stats[Symbol(:Xsectσ, Fy)] =  sqrt(unorderedE[Symbol(:sum2, Fy)]/stats[:S] - (unorderedE[Symbol(:sum,Fy)]/stats[:S])^2)
          stats[Symbol(:Xsectσ, Fy, :_scaled2x)] = stats[Symbol(:Xsectσ, Fy)]/sqrt(stats[:Δt])
          stats[Symbol(:Xsectσ, Fy, :_scaled2xsqrt12)] = stats[Symbol(:Xsectσ, Fy, :_scaled2x)] *12^0.5        

          if :σ2yε_calc ∈ uniqueparams
            stats[Symbol(Fstat, Fy, :_r2_brss)] = rss/(rss + unorderedE[:σ2yε_calc]*stats[:S])
          end

          if :σ2yε_calc2 ∈ uniqueparams
            stats[Symbol(Fstat, Fy, :_r2_brss2)] = rss/(rss + unorderedE[:σ2yε_calc2]*stats[:S])
          end
        end
      end #end block on CVR2

      # a different type of R² meansure
      for Fx ∈ intersect(uniqueparams, [:xβ, :xF, ])
        if Fx ∈ dchain.paramgroup && :x ∈ dchain.paramgroup
          for Fstat ∈ Fpredictivestats
            x = dchain[dchain.paramgroup .== :x, Fstat]
            xhat = dchain[dchain.paramgroup .== Fx, Fstat]

            stats[Symbol(Fstat, Fx, :_r2)] = cor(x, xhat)^2
            tss = sum((x .- mean(x)).^2)
            rss = sum((xhat .- mean(x)).^2)
            stats[Symbol(Fstat, Fx, :_r2_rss)] = rss/tss
            stats[Symbol(Fstat, Fx, :_r2_res)] = 1-sum((x .- xhat).^2)/tss


            stats[Symbol(Fstat, Fx,  :_te)] = (xhat .- x).^2 |> mean |> sqrt

            stats[Symbol(:Xsectσ, Fx)] =  sqrt(unorderedE[Symbol(:sum2,Fx)]/stats[:T] - (unorderedE[Symbol(:sum,Fx)]/stats[:T])^2)
            stats[Symbol(:Xsectσ, Fx, :_scaledsqrt12)] =  stats[Symbol(:Xsectσ, Fx)] * 12^0.5

            #a crude approach to accounting for the measurement error in the x R^2 to make it more comparable to the y R2
            if :τy ∈ uniqueparams
              τy = dchain[dchain.paramgroup .== :τy,:E][begin]
              σ2εy = 1/τy/stats[:Δt]
              stats[Symbol(Fstat, Fx, :_r2adj)] = cov(x, xhat)^2/(var(xhat)*(var(x) + σ2εy))
            end

            if :σ2xε_calc ∈ uniqueparams
              stats[Symbol(Fstat, Fx, :_r2_brss)] = rss/(rss + unorderedE[:σ2xε_calc]*stats[:T])
              stats[Symbol(Fstat, Fx, :_r2_brssadj)] = rss/(rss + unorderedE[:σ2xε_calc_adj]*stats[:T])

            end
            if :σ2xε_calc2 ∈ uniqueparams
              stats[Symbol(Fstat, Fx, :_r2_brss2)] = rss/(rss + unorderedE[:σ2xε_calc2]*stats[:T])
              stats[Symbol(Fstat, Fx, :_r2_brss2adj)] = rss/(rss + unorderedE[:σ2xε_calc2_adj]*stats[:T])
            end
          end
        end
      end

      #now summarize the aggregate stats for each parameter
      for Fstat ∈ Faggregatestats
        unorderedstats = Dict(dchain[!, :param] .=> dchain[!, Fstat])
        for param ∈ dchain.param
          nt = unorderednotes[param]
          if nt≡missing
            colname = Symbol(Fstat, :_, param)
          else
            colname = Symbol(Fstat, unorderedparamgroups[param], :_, unorderednotes[param])
            maybedate = tryparse(Date, "$nt")
            if maybedate  !== nothing
              dateindex[colname] = maybedate
            end
            
          end
          paramgroupindex[colname] = unorderedparamgroups[param]

          if haskey(stats,colname)
            @eval Main stats=$stats
            throw("Duplicate key $colname found in stats table")
            
          end
          stats[colname] = unorderedstats[param]
        end
      end
      
      #record the priors
      priorstring = map(d.param[d.prior.≢missing], d.prior[d.prior .≢ missing]) do Fθ,prior
        "$Fθ=>$prior"
      end
      stats[:prior] = join(priorstring,", ")

      agg = vcat(agg, DataFrame(stats), cols=:union)
    end
  end
  

  agg.source = ((f,c)->Symbol(f,:_, c)).(agg.fund, agg.chain)
  aggt = permutedims(agg, :source)

  #preserve paramgroup ordering, but we want to sort on date within paramgroups
  aggt.paramgroup = aggt.source .|> Symbol .|> s->haskey(paramgroupindex,s) ? paramgroupindex[s] : s
  paramgroupidindex = Dict(pg =>k for (k,pg) ∈ enumerate(aggt.paramgroup |> unique))
  aggt.paramgroupid = aggt.paramgroup .|> pg->paramgroupidindex[pg]

  #@eval Main paramgroupidindex=$paramgroupidindex

  aggt = [DataFrame(date=(aggt.source .|> Symbol .|> s->haskey(dateindex, s) ? dateindex[s] : missing)) aggt]
  sort!(aggt, [:paramgroupid,:date])
  select!(aggt, Not([:paramgroup,	:paramgroupid]))
  aggt |> CSV.write("$analysispath/$rid/$(rid)_aggt.csv", bom=true)

  

end

#wrapper function to write out the summary file
function createsummaryfile(;
  dgp,
  rid,
  outputpath,
  runsplit = PARAM[:iosummaryrunsplit])

  if runsplit
    splitstats = summarizechainoutput(; dgp,splitchains=true)
    splitstats |> CSV.write("$outputpath/summary/split/s_$rid.csv", bom=true)
  else
    splitstats=nothing
  end

  wholestats = summarizechainoutput(; dgp,splitchains=false)
  wholestats |> CSV.write("$outputpath/summary/whole/s_$rid.csv", bom=true)

  return (; splitstats, wholestats)


end

#this tests the bootstrap
function testsimplebootstrap(; 
    drawtype=:ma,
    truedraws=10^4, 
    bootstrapattempts=10^3, 
    bootstrapdraws=10^3,
    T=500, 
    lags=3, 
    minblocklag=1,
    maxblocklag=100,
    acceptableautocor=0.05,
    σ2ε=1.0,
    blocksize=nothing,)

    local drawfunc::Function


    function drawma(ϕ;T=T,σ2ε=1.0, )
      θraw = rand(Normal(1.0,σ2ε), T+lags)
      θ = reduce(hcat,[θraw[(1+l):(end-(lags-l))] for l ∈ 0:lags])*ϕ
      θ
    end

    function drawar1(φ;T=T,σ2ε=1.0)
      @assert lags ==1 "only lags=1is supported for AR model"
      θ = rand(Normal(1.0,σ2ε), T)
      θ[1] = σ2ε/(1-φ^2)
      for t ∈ 2:T
        θ[t] += θ[t-1] * φ
      end
      θ
    end



    if drawtype == :ma
      ϕ = (1+lags):-1:1 |> collect |> ϕraw->ϕraw ./ sum(ϕraw)

      θbigT = drawma(ϕ;T=10^4)
      @assert ((ceil((lags) / 2))*2+1 ==
        findlowcorlag(θbigT; 
          minlag=minblocklag, 
          maxlag=maxblocklag, 
          acceptableautocor, 
          testfindlowcorlag=true).lag+1) "
        findlowcorlag(θ0; ...)= $(findlowcorlag(θbigT; 
          minlag=minblocklag, 
          maxlag=maxblocklag, 
            acceptableautocor, 
            testfindlowcorlag=true).lag+1), should be
        $((ceil((lags) / 2))*2+1 )"

      resultsactualemp = [mean(drawma(ϕ)) for i ∈ 1:truedraws]
      seactualemp = std(resultsactualemp)

      #derive the analytical value
      #the formula as worked out in the dimson beta justification doc is σ^2ε*sum_{q∈0:(malags-Δ)}(ϕ^q*ϕ^(q+Δ)) s.t. Δ=abs(i-j)
      Σ = [abs(i-j) |> Δ->ifelse(Δ≤lags, dot(ϕ[1:(end-Δ)], ϕ[(Δ+1):end]), 0.0) for i ∈ 1:T, j ∈ 1:T]
      seactualana = ((sum(Σ))/T^2)^0.5

      drawfunc=(; kwargs...)->drawma(ϕ; kwargs...)
    elseif drawtype == :ar1
      φ = 0.6
      θbigT = drawar1(φ;T=10^4)
      rawblocknumber = ceil(log(acceptableautocor)/log(φ))
      targetblocknumber = max(ceil(rawblocknumber/max(1,minblocklag ÷ 2))*max(1,minblocklag ÷ 2), minblocklag)
      @assert (findlowcorlag(θbigT; 
          minlag=minblocklag, 
          maxlag=maxblocklag, 
          acceptableautocor, 
          testfindlowcorlag=true).lag + 1
        ≥ max(minblocklag, targetblocknumber - max(1,minblocklag ÷ 2))) "
        findlowcorlag(.)=
        $(findlowcorlag(θbigT; 
          minlag=minblocklag, 
          maxlag=maxblocklag, 
          acceptableautocor, 
          testfindlowcorlag=true).lag+1), should be
        $(ceil(log(acceptableautocor)/log(φ)))"

      resultsactualemp = [mean(drawar1(φ)) for i ∈ 1:truedraws]
      seactualemp = std(resultsactualemp)
      Σ = [φ^abs(i-j) for i ∈ 1:T, j ∈ 1:T] .* σ2ε/(1-φ^2)
      w = ones(T)/T
      seactualana = (w'Σ*w)^0.5
      drawfunc =(; kwargs...)->drawar1(φ; kwargs...)
    end

    bootstrapseresults = [drawfunc(;T) |>
        θ-> simpleblockbootstrap(θ, mean;
        blocksize=something(blocksize, 
          findlowcorlag(θ; 
            minlag=minblocklag, 
            maxlag=maxblocklag, 
            acceptableautocor, 
            lagmultiplier=2.0).lag+1), 
        numdraws=bootstrapdraws) |> 
          r->std(r) for i ∈ 1:bootstrapattempts]


    @info "MC se = $seactualemp"
    @info "Analytical se = $seactualana"
    @info "Mean BS se=$(mean(bootstrapseresults)) with std $(std(bootstrapseresults))"


end
