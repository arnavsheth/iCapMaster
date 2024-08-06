#=function analyzechain(; resultid, chainid, outputpath,
    analysispath=PARAM[:analysispath],
    )

  #create a directory to hold anlsyis results
  resultid = "rid$(Dates.format(now(),"yymmdd-HHMM"))"
  outputpath = "$analysispath/$resultid"    
end=#


function formanalysispath(; rid,
  analysispath=PARAM[:analysispath],)
  outputpath = "$analysispath/$rid"

  @assert !ispath(outputpath) "
    Path $outputpath already exists! Please wait 1 minute from the beginning of the past run (or delete the old path)"
  
  mkdir(outputpath)
  mkdir("$outputpath/chain")
  mkdir("$outputpath/fig")
  mkdir("$outputpath/tab")
  mkdir("$outputpath/archive") 
  mkdir("$outputpath/summary") 
  mkdir("$outputpath/summary/whole")
  mkdir("$outputpath/summary/split")

  #store the parameter file with the results
  cp(PARAMETER_FILE, "$outputpath/$(rid)_Parameters.xlsx")

  return outputpath
end

function chainanalysis(; dgp::DGP, 
    analysispath=PARAM[:analysispath],
    label=PARAM[:mcmcchainanalysislabel], 
    outbinstream = OUT_BIN_STREAM,
    recordrawdata=PARAM[:iorecordrawdata],
    mcmcruntype=PARAM[:mcmcruntype],
    stoprule,
    maxrecords,)
  
  #create a directory to hold anlsyis results
  rid = "$(label)_rid$(Dates.format(now(),"yymmdd-HHMM"))"

  outputpath = formanalysispath(;rid)

  try
    dgp = mcmc(mcmcruntype, dgp; maxrecords, stoprule)

    dgp = addderived(; dgp, )

    @unpack records=dgp
     
    recordrawdata && outbinstream("$outputpath/chain/$rid.zstd", dgp)


  catch err
    mv(outputpath, "$(analysispath)/ERR_$rid")
    #show(stdout, MIME"text/plain"(), stacktrace(catch_backtrace()))
    @error "MCMC failed" exception=(err, catch_backtrace())
    throw("MCMC failed")
  end

  return (;dgp, rid, outputpath)
end

simulatechainanalysis(;dgp=loadmcmctest()) = chainanalysis(;
  dgp, stoprule=PARAM[:simulatestoprule], 
  maxrecords=PARAM[:simulatemaxrecords],)

#orders the funds in a manner that resolves prior-related dependencies
function resolvepriorlinkindex(;priorlinkindex, Fassetreturns)
  allassets = unique!([values(priorlinkindex) |> collect; keys(priorlinkindex)|> collect])
  #@assert setdiff()



  resolvedassets=Symbol[]
  #recursively resolve the dependencies
  function resolveasset!(f::Symbol, resolved)
    ((f ∈ resolved) || (f ∉ allassets) ) && return resolved
    (f ∈ keys(priorlinkindex)) && resolveasset!(priorlinkindex[f], resolved)
    push!(resolved, f)
  end

  for f ∈ Fassetreturns
    resolvedassets = resolveasset!(f,resolvedassets)
  end

  isempty(resolvedassets) && throw("no assets with historical priors found in selected assets
  (check iobatchassetstoanalyze and priorlinkindex for misspellings")
  for f ∈ allassets
    if haskey(priorlinkindex, f)
      @assert priorlinkindex[f] ∈ allassets "Dependency $(priorlinkindex[f]) not found in allassets"
    end
  end

  @assert issetequal(resolvedassets, allassets) "
    resolvedassets=$resolvedassets but allassets=$allassets and Fassetreturns=$Fassetreturns"

  return resolvedassets
end

function batchchainanalysis(; 
  assetstoanalyze::T=PARAM[:iobatchassetstoanalyze], 
    setseed=nothing,
    priorset=PARAM[:livepriorset],
    priorlinkindex=haskey(PARAM[priorset], :priorlinkindex) ? PARAM[priorset][:priorlinkindex] : nothing,
    ) where T



  assets=loadassets()
  if T<:Symbol && assetstoanalyze==:all
    #loading the input
    Fassetreturns = setdiff(propertynames(assets), [:date])
  elseif T<: Vector{Symbol}
    Fassetreturns = assetstoanalyze
    @assert issetequal(Fassetreturns, intersect(Fassetreturns, propertynames(assets))) "
      $(setdiff(Fassetreturns,propertynames(assets))) not found in assets file.
      Asset file names: $(propertynames(assets))"
  else
    throw("Unrecognized assetstoanalyze $assetstoanalyze")
  end

  if priorlinkindex !== nothing
    Fassetreturns = resolvepriorlinkindex(;priorlinkindex, Fassetreturns)
  end

  if setseed !== nothing
    Random.seed!(setseed)
  end

  return batchchainanalysis(Fassetreturns)
end

function batchchainanalysis(Fassetreturns::Vector{Symbol}; 
    recordrawdata=PARAM[:iobatchrecordrawdata],
    label=PARAM[:batchmcmcchainanalysislabel], 
    mcmcruntype=PARAM[:mcmcruntype],
    analysispath=PARAM[:analysispath],
    maxrecords=PARAM[:livemaxrecords],
    stoprule=PARAM[:livestoprule],
    runanalysis=PARAM[:runanalysis],
  )

  @unpack aggregatefile, bandplots = runanalysis
  #create a directory to hold anlsyis results
  rid = "$(label)_rid$(Dates.format(now(),"yymmdd-HHMM"))"

  outputpath = formanalysispath(;rid)


  for (i, Fassetreturn) ∈ enumerate(Fassetreturns)
    try
      println("Analyzing asset $i/$(length(Fassetreturns)): $Fassetreturn")
      dgp = loadliveDGP(Fassetreturn; rid)
      dgp = mcmc(mcmcruntype, dgp; stoprule, maxrecords)
      #@unpack records=dgp
      

      dgp = addderived(; dgp,)
      
      #@eval Main dgp = $dgp
      fundrid = "$(dgp.records.chainid)_$(rid)"
      recordrawdata && outbinstream("$outputpath/chain/$fundrid.zstd", dgp)
      analyzemcmcoutput(;unburnt=dgp, rid=fundrid, outputpath)
  
  
    catch err
      mv(outputpath, "$(analysispath)/ERR_$rid")
      #show(stdout, MIME"text/plain"(), stacktrace(catch_backtrace()))
      @error "MCMC failed" exception=(err, catch_backtrace())
      throw("MCMC failed")
    end   
  end

  aggregatefile &&  createaggregatesummaryfile(;rid)
  bandplots && createbandplots(; rid)


  return nothing

end


function writedataascsv(; 
    rid=PARAM[:iorid], 
    inbinstream=IN_BIN_STREAM,
    analysispath=PARAM[:analysispath],
    burnchain=true)

  outputpath = "$analysispath/$rid"
  dgp = "$analysispath/$rid/chain/$rid.zstd" |> inbinstream
  @unpack dims = dgp

  if burnchain
    records = burnchainrecords(; records, dgp)
    chaindf = records.chain |> DataFrame
  else #here we record both the burn-in and the non-burn-in as a CSV, but include a label
    records = dgp.records
    chaindf = records.chain |> DataFrame
    chaindf.burnin = falses(nrow(chaindf)) |> Vector{Bool}
    chaindf[chaindf.iteration ≤ dims.numburnrecords] .= true
  end

  chaindf = records.chain |> DataFrame
  if nrow(chaindf) ≤ 100_000
    chaindf |> CSV.write("$outputpath/chain/$(rid).csv")
  elseif recordrawdatacsv
    @warn "CSV not recorded because nrow(chaindf)=$(nrow(chaindf))>100_000"
  end

  return nothing
end

function analyzemcmcoutput(;
    unburnt=loadrecords(;),
    burnt = DGP(unburnt, records=burnchainrecords(;dgp=unburnt,), strict=false),
    runanalysis=PARAM[:runanalysis], 
    abortonanalysisfailure=PARAM[:diagabortonanalysisfailure],
    rid=PARAM[:iorid],
    outputpath="$(PARAM[:analysispath])/$rid",)
  
  try  
    @assert burnt.records.burnt == true == (!unburnt.records.burnt)
    @unpack dims=unburnt
    if runanalysis.cv 
      burnt = crossvalidatedr2s(burnt, numburnrecords=unburnt.records.numburnrecords)
    end


    runanalysis.summaryfile && createsummaryfile(;dgp=burnt, rid, outputpath)
    runanalysis.cannedplots && cannedplots(;records=burnt.records, dims, rid, outputpath)
    runanalysis.analysisplots && analysisplots(;dgp=burnt, rid, outputpath)
    runanalysis.convergenceplots && parameterconvergence(;records=unburnt.records,dims, rid, outputpath)

  catch err
    @eval Main unburnt=$unburnt
    @eval Main burnt=$burnt
    msg = "Analysis failed for chainid $(burnt.records.chainid) with error $err. 
    burnt and unburnt dgp saved to REPL. "

    abortonanalysisfailure && error(msg)
    @warn(msg)
  end



   
  return nothing
end
