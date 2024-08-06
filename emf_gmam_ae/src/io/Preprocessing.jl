

#########functions related to reading in factors#########
function processfactors(;
    rawfilename = PARAM[:factorrawfilename],
    rawiswide = PARAM[:factorrawiswide],
    longfields = PARAM[:factorrawlongfields],
    usecommondatesonly::Bool = PARAM[:factorusecommondatesonly],
    Frf::Symbol = PARAM[:factorFrf],
    rescale = PARAM[:factorrescale,],
    rawdateformat = PARAM[:factordateformat],
    saveas = PARAM[:factorsaveas],
    processeddatapath = PARAM[:processeddatapath],
    excess = PARAM[:factorexcess],
    changetolog=PARAM[:factorchangetolog],
    frequency::Symbol = PARAM[:factorfrequency],)



  fwide = processrawdata(; usecommondatesonly, rawdateformat, rescale, 
    rawfilename, rawiswide, longfields, frequency, 
    changetolog, colprefix=:F_)

  #subtract the rfr from all returns that are not excess
  Ffactors = setdiff(propertynames(fwide), [:date; Frf;])
  if (typeof(excess) <: Bool) && (!excess)
    fwide[:,Ffactors] .-= fwide[!,Frf]
  elseif typeof(excess) <: Bool
    @assert excess
  else
    for Ffactor ∈ Ffactors
      if !excess[Ffactor]
        fwide[:, Ffactor] .-= fwide[!, Frf]
      end
    end
  end

  #write out the processed data file
  fwide |> CSV.write("$processeddatapath/$saveas.csv")

  return fwide
end


function processassets(;
  rawfilename = PARAM[:assetrawfilename],
  rawiswide = PARAM[:assetrawiswide],
  longfields = PARAM[:assetrawlongfields],
  usecommondatesonly::Bool = PARAM[:assetusecommondatesonly],
  rescale = PARAM[:assetrescale,],
  rawdateformat = PARAM[:assetdateformat],
  saveas = PARAM[:assetsaveas],
  processeddatapath = PARAM[:processeddatapath],
  changetolog=PARAM[:assetchangetolog],
  frequency::Symbol = PARAM[:assetfrequency],
  converttofrequency = PARAM[:assetconverttofrequency])

  awide = processrawdata(; usecommondatesonly, rawdateformat, rescale, 
    rawfilename, rawiswide, longfields, frequency, 
    changetolog, colprefix=:A_)
  
  awide = summarizefrequency(awide; sourcefreq=frequency, targetfreq=converttofrequency)



  #write out the processed data file
  awide |> CSV.write("$processeddatapath/$saveas.csv")   
  
  return awide

end


function processrawdata(;
    rawfilename,
    rawiswide,
    usecommondatesonly::Bool,
    Ffieldstokeep=nothing,
    rescale::Trescale,
    rawdateformat::DateFormat,
    frequency::Symbol,
    colprefix::Symbol,
    rawvalidatedates = true,
    datapath = PARAM[:datapath],
    mindate::Union{Nothing, Date} = PARAM[:datamindate],
    maxdate::Union{Nothing, Date} = PARAM[:datamaxdate],
    longfields=nothing,
    changetolog
  ) where {Trescale}

  d = CSV.File("$datapath/$rawfilename.csv") |> DataFrame

  if  !(eltype(d.date) <: Union{Missing, Date})
    d.date = d.date .|> (i->Date("$i", rawdateformat)) .|> fix2digityear
  end
  d.date .= d.date .|> lastdayofmonth
  if rawiswide
    dwide = d
  else
    @unpack variable, value = longfields
    dlong = @view(d[:,[:date,variable,value]])

    if sum(nonunique(dlong)) ≠ 0
      @eval Main dlong=$dlong
      throw("non-unique date-fund pairs found")
    end
    dwide = unstack(dlong, variable, value)
  end




  mindate = mindate ≡ nothing ? mindate : firstdayofmonth(mindate)
  maxdate = maxdate ≡ nothing ? maxdate : lastdayofmonth(maxdate)


  rename!(dwide, ["date"; names(dwide)[2:end] .|> standardizename .|> s->string(colprefix,s)])



  if !allunique(dwide.date)
    @eval Main dwide=$dwide
    throw("dates are not unique in wide df")
  end
  sort!(dwide, :date)

  #ensure we operate on the desired date range
  dwide = dwide[something(mindate, Date(1,1,1)) .≤ dwide.date .≤ something(maxdate, Date(9999,1,1)),:]


  #only capture the relevant factors

  Fdata = Ffieldstokeep !== nothing ? Ffieldstokeep : setdiff(propertynames(dwide), [:date, ])
  dwide = dwide[:,[:date; Fdata]]

  #scale depending in case the data is in percentage points etc
  if Trescale <: Float64
    rescaleindex::Dict{Symbol, Float64} = Dict(Fdata .=>  rescale)
  else
    rescaleindex = rescale
  end

  @eval Main dwide = $dwide
  for Fc ∈ Fdata
    dwide[!,Fc] ./= rescaleindex[Fc]
  end

  if usecommondatesonly

    dwide = dwide[completecases(dwide, Fdata),:]

  #else #WARNING- may want to change imputation method
  #uncomment and implement if we need imputation
  #  dwide .= imputebyrow(dwide, Val{:mean}())
  end

  if (abs.(dwide[!,Fdata] |> Matrix{Union{Missing,Float64}})) |> skipmissing |> mean > 0.25
    throw("data has extremely wide mean of $(mean(abs.(dwide[!,Fdata])))
      Something is probably wrong with the scaling- use extreme caution before disabling this check")
  end

  if typeof(changetolog) <: AbstractVector
    fieldstolog=changetolog
  elseif changetolog
    fieldstolog=setdiff(propertynames(dwide),[:date])
  elseif !changetolog
    fieldstolog=Symbol[]
  else
    @assert false
  end

  for f ∈ fieldstolog
    dwide[:,f] .= (1 .+ dwide[:,f]) .|> log
  end


  (mindate !== nothing) && (@assert mindate ≤ firstdayofmonth(minimum(dwide.date)))
  (maxdate !== nothing) && (@assert maxdate ≥ lastdayofmonth(maximum(dwide.date)))
  for f ∈ Fdata
    try
      validatedates(dwide[completecases(@view(dwide[:,[:date,f]])),:date]; validateexactfrequency=rawvalidatedates, frequency)
    catch err
      @error "Validate failed for f=$f"
      throw(err)
    end
  end



  return dwide
end


#simple stub to load the asset data
function loadassets(;
  assetfile=PARAM[:formattedassetfile],
  processeddatapath = PARAM[:processeddatapath],)

  assets = CSV.File("$processeddatapath/$assetfile.csv") |> DataFrame
  if !(Date <: eltype(assets.date))
    assets.date = assets.date .|> dt->Date(dt, dateformat"yyyy-mm-dd")
  end

  return assets
end


#simple stub to load the factor data
function loadfactors(;
  factorfile=PARAM[:formattedfactorfile],
  processeddatapath = PARAM[:processeddatapath],
  modelfactors,
  Frf,
  factorfrequency,
  )
  
  factors=CSV.File("$processeddatapath/$factorfile.csv") |> DataFrame

  if !(Date <: eltype(factors.date))
    factors.date = factors.date .|> dt->Date(dt, dateformat"yyyy-mm-dd")
  end

  @assert issorted(factors.date)
  validatedates(factors.date, frequency=factorfrequency, validateexactfrequency=true)

  #keep only the correct factors
  if modelfactors ≡ :all
    Ffactors = setdiff(propertynames(factors), [:date, Frf])
  elseif typeof(modelfactors) <: AbstractVector
    select!(factors, [ :date; Frf; modelfactors])
    Ffactors=modelfactors
  end

  factors=factors[completecases(factors, [:date;Frf; Ffactors]),:]
  return (;Ffactors, factors) 

end

function extendfactordata(dgp;  
  modelfactors,
  Frf,
  factorfrequency,
  tlocalparams=PARAM[:iotlocalparams_full], 
  slocalparams=PARAM[:ioslocalparams_full])

  #@warn "UNFINISHED!! "

  @unpack dims = dgp
  @unpack Fθ2name, K,P, Δt, priorset = dims
  @assert (factorfrequency≡:month) "loadfactordata only supports monthly factor frequencies"


  baseydate = Fθ2name[Symbol("y[1]")]
  possibleydates = (vcat((baseydate - Month(Δt)):Month(-Δt):(baseydate - Year(200)), 
    baseydate:Month(Δt):(baseydate + Year(200))) .|> lastdayofmonth) |> sort!
  
  @unpack Ffactors, factors = loadfactors(;  modelfactors, Frf,  factorfrequency,)


  #here we want to maximize the number of y dates given the factor dates
  ydates = intersect(factors.date[(Δt+P):end],possibleydates)
  @assert issorted(ydates)
  Δydates = setdiff(1:(dims.S) .|> s->Fθ2name[Symbol("y[$s]")],ydates )
  @assert Δydates  |> isempty "Δydates: $Δydates"

  validatedates(ydates, frequency=Dict(1=>:month, 3=>:quarter, 12=>:year)[Δt], validateexactfrequency=true)
  
  #select the factordates and validate integrity of date series
  factordates = selectfactordates(; P, Δt, ydates, factorfrequency, 
    availablefactordates=factors.date)
  factors_full = sort!(innerjoin(factors, DataFrame(date=factordates), on=:date),:date)
  @assert setdiff(1:(dims.S) .|> s->Fθ2name[Symbol("y[$s]")],ydates ) |> isempty

  S = length(ydates)
  #create the modified objects
  dims_full = Dims(;S,P,K, Δt, dates=factors_full.date, priorset=nothing,
    slocalparams, tlocalparams=Symbol[], addxylabels=false)
  tstart_full::Int = findfirst(==(dims.dates[begin]), dims_full.dates)
  F = [ones(size(factors_full,1)) factors_full[!, Ffactors] |> Matrix{Float64}]
  r = factors_full[!, Frf]
  data_full=FrData(;F, r)

  #check we got the dimensionality right
  if Δt==1
    @assert cov([F r]) ≈ (
      [ones(dims_full.T) (factors_full[:, [Ffactors; Frf]] |> Matrix{Float64})] |> cov)
  else
    @assert cov([F r]) ≈ (
      [ones(dims_full.T) (
      sort!(innerjoin(factors, factors_full[:, [:date]], on=:date))[:, [Ffactors; Frf]] |> Matrix{Float64})] |> cov)
  end

  ydates_asif_mo = factors.date[(1+P):end]
  factordates_asif_mo = selectfactordates(; P, Δt=1, ydates=ydates_asif_mo, factorfrequency, 
  availablefactordates=factors.date)
  @assert all(factordates_asif_mo .== factors.date)
  dims_full_asif_mo = Dims(;S=length(ydates_asif_mo),P,K, Δt=1, 
    dates=factordates_asif_mo, tlocalparams,slocalparams=Symbol[], addxylabels=false) 
  tstart_full_asif_mo::Int = findfirst(==(dims.dates[begin]), dims_full_asif_mo.dates)
  data_full_asif_mo=FrData(;
    F = [ones(size(factors,1)) factors[!, Ffactors] |> Matrix{Float64}], 
    r = factors[!, Frf])

  return(; dims_full, data_full, tstart_full,
    dims_full_asif_mo ,data_full_asif_mo, tstart_full_asif_mo)
end

function selectfactordates(; P, Δt, ydates, availablefactordates, factorfrequency)
  @assert factorfrequency ≡ :month "factorfrequency other than month is not supported"
  firstydate, lastydate = minimum(ydates), maximum(ydates)
  factordates = (((firstydate - Month(P+Δt-1)):Month(1):lastydate) |> collect) .|> lastdayofmonth
  validatedates(factordates, frequency=factorfrequency, validateexactfrequency=true)
  factordates = intersect(factordates, availablefactordates)
  return factordates
end



function loadliveDGP(  Fassetreturn::Symbol; 
  rid,
  assetfrequency=PARAM[:assetfrequency],
  P::Int=PARAM[:livenummalagsforassetfrequency][assetfrequency],
  defaultpriorset=PARAM[:livepriorset],
  priorsetoverrides=PARAM[:livepriorsetoverrides],
  numsamplerecords=PARAM[:livenumsamplerecords], 
  numburnrecords=PARAM[:livenumburnrecords], 
  numchains=PARAM[:livenumchains],
  fieldstocapture=PARAM[:livefieldstocapture],
  mcmcinitmethod=PARAM[:defmcmcinitmethod],
  model=MODEL_INDEX[PARAM[:livemodel]],
  modelfactors=PARAM[:livemodelfactors],
  Frf = PARAM[:factorFrf],
  factorfrequency=PARAM[:factorfrequency],
  )

  priorset =  (haskey(priorsetoverrides, Fassetreturn) ? 
    priorsetoverrides[Fassetreturn] : defaultpriorset)
    

  #get the asset and factor returns
  assets = loadassets()
  select!(assets, [:date; Fassetreturn])
  assets=assets[completecases(assets),:]
  @assert issorted(assets.date)
  validatedates(assets.date, frequency=assetfrequency, validateexactfrequency=true)
  
  @unpack Ffactors, factors = loadfactors(;  modelfactors, Frf,  factorfrequency,)



  K=length(Ffactors)+1

  #check month dates are aligned
  @assert all(factors.date .== lastdayofmonth.(factors.date))
  @assert all(assets.date .== lastdayofmonth.(assets.date))

  

  #the simplest case
  if factorfrequency≡:month
    # this is where the "delta t" variable gets defined
    Δt = Dict(:month=>1,:quarter=>3,:year=>12)[assetfrequency]


    #select the usable ydates and validate integrity of date series
    ydates = intersect(factors.date[(Δt+P):end],assets.date)
    @assert issorted(ydates)
    validatedates(ydates, frequency=assetfrequency, validateexactfrequency=true)
    @assert issetequal(ydates, intersect(ydates, assets.date))
    assets = sort!(innerjoin(assets, DataFrame(date=ydates), on=:date),:date)

    #select the factordates and validate integrity of date series
    factordates = selectfactordates(; P, Δt, ydates, factorfrequency, 
      availablefactordates=factors.date)
    factors = sort!(innerjoin(factors, DataFrame(date=factordates), on=:date),:date)

  elseif factorfrequency≡:quarter
    throw("TODO:  factorfrequency≡:quarter")

  elseif factorfrequency≡:year
    throw("TODO:  factorfrequency≡:year")

  else
    throw("unsupported factorfrequency $factorfrequency")
  end

  #form the dims object
  S = length(assets.date)
  @eval Main dates=$(factors.date)
  
  dims=Dims(; K,S,P,Δt, dates=factors.date, 
    Fβs=[:intercept; Ffactors .|> Ffactor->replace(string(Ffactor), "F_"=>"") |> Symbol],
    priorset)
  #  dims=Dims(; K,S,P,Δt, dates=factors.date, 
  #Fβs=[:β1intercept; ((i,Ffactor)->Symbol(:β, i, Ffactor)).(2:K, Ffactors)...;])
  @assert dims.T==length(factors.date)

  #form the data object
  r = factors[!,Frf]
  F = [ones(size(factors,1)) factors[!, Ffactors] |> Matrix{Float64}]
  y= assets[!,Fassetreturn] .|> Float64
  data=Data(;F, y, r)

  #get the priors
  @info "Loading prior set $(priorset) for model $model"
  hyper=loadpriors(priorset, model; dims, data, fundname=Fassetreturn, rid)

  #initialize the mcmc
  Θ=initmcmc(mcmcinitmethod, model;dims,data,hyper)
  records = ChainRecords(fieldstocapture, ; chainid=Fassetreturn, Θ, numsamplerecords, numburnrecords, numchains, )

  dgp= DGP(;Θ,data, dims, hyper, records)

  return dgp

  

end





