
#launches the stopping rule
checkstoprule(stoprule::Symbol; kwargs...) = checkstoprule(Val{stoprule}(); kwargs...)
function checkstoprule(Tstoprule::Val; dgp, kwargs...)
  @assert all(dgp.records.numrecords .== dgp.records.lastrecord)

  return checkstoprule(Tstoprule, dgp; kwargs...)
end

checkstoprule(::Val{:fixed}, dgp; maxrecords=PARAM[:stopfixed], kwargs...)  = (;
    stop=all(dgp.records.lastrecord .≥ maxrecords), 
    heuristicexpandfrac=maxrecords/minimum(dgp.records.lastrecord)-1.)

checkstoprule(::Val{:never}, args...;kwargs...) = (;stop=false)
checkstoprule(::Val{:always}, args...;kwargs...) = (;stop=true)


checkstoprule(V::Val{:bootstrapess}, dgp::DGP{<:DGPMeanModelParameters}; 
  bootstrapfocalfields=[:μ, :τ],
  kwargs...) = _checkstoprule(V, dgp; bootstrapfocalfields, kwargs...)
  
checkstoprule(V::Val{:bootstrapess}, dgp::DGP; kwargs...) = _checkstoprule(V, dgp; kwargs...)

#this imposes a bootstrap-based ess
function _checkstoprule(::Val{:bootstrapess}, dgp; 
    miness = PARAM[:stopbootstrapminess],
    maxblocklag::Int=round(dgp.records.numsamplerecords * PARAM[:stopbootstrapmaxblocksizefrac]) |> Int,
    minblocklag::Int=PARAM[:stopbootstrapminblocksize],
    chainsummationmethod=PARAM[:stopchainsummationmethod],
    acceptableautocor=PARAM[:bootstrapacceptableautocor],
    bootstrapfocalfields=PARAM[:stopbootstrapfocalfields],
    lagmultiplier=PARAM[:bootstraplagmultiplier],
    verbose = PARAM[:stopverbose],
    maxincrease = PARAM[:sampleheuristicmaxincrease],
    heuristicslack = PARAM[:sampleheuristicslack],
    kwargs...
    )

  records=burnchainrecords(;dgp)

  @unpack numchains,numsamplerecords, chain, lastrecord,chainfieldindex = records


  Ffocals= expandchainfields(records, bootstrapfocalfields)
  stopesss = Dict{Symbol, Any}()
  laglengthfound = Dict{Symbol,Any}(Fθ=>false for Fθ ∈ Ffocals)
  stopresults = Dict{Symbol,Bool}()
  info = IOBuffer()

  for Fθ ∈ Ffocals
    θmat = chain[:,Fθ,:] |> Matrix{Float64}
    @assert (numsamplerecords, numchains) ≡ size(θmat)
    θσ2s = Float64[]


    for (c,θ) ∈ enumerate(eachcol(θmat))
      blocksize, lagautocor, acceptablelagfound = findlowcorlag(θ; 
        minlag=minblocklag, 
        maxlag=maxblocklag, 
        acceptableautocor, 
        lagmultiplier) |> nt->(nt.lag+1, nt.lagautocor, nt.acceptablelagfound)
      
      laglengthfound[Fθ] |= acceptablelagfound
      push!(θσ2s,var(simpleblockbootstrap(θ,mean; blocksize)))
      verbose && print(info, 
        "$(Symbol(Fθ,"_", c)): blocksize=$blocksize, lagautocor=$lagautocor, θσ=$(θσ2s[end]), ess=$(var(θ)/θσ2s[end])\n")

    end

    ess = chainsummationmethod((eachcol(θmat) .|> var) ./ θσ2s)
    #=if length(esss)==1
      ess = esss[begin]
    elseif chainsummationmethod == :sum
      ess = esss |> sum
    elseif chainsummationmethod == :min=#
    #the only scenario where a NaN should be present is if all the same values are drawn
    ess≡NaN && (@warn "NaN value for $Fθ. Its probably ok; but that said,probably shouldn't use γ for this test.")
    
    stopesss[Fθ] = ess
    stopresults[Fθ] = laglengthfound[Fθ] && (ess ≡ NaN || ess≥miness)
    verbose && (numchains > 1) && print(info, "$(Symbol(Fθ,"_tot: ess=$(ess[end]), stop=$(ess≡NaN || ess≥ miness)"))\n")
  end

  verbose && @info(String(take!(info)))

  stop = stopresults |> values |> all

  if stop
    verbose && println("Convergence test passed with $numsamplerecords samples")
    heuristicexpandfrac=nothing
  else
    #take a guess at the number of additional records needed
    #hard-coded, but its a heuristic
    heuristicexpandfrac = min(maxincrease,(miness/minimum(values(stopesss))-1)*(1+heuristicslack))

    #if a lag length is not found, expand a minimum of 100%
    heuristicexpandfrac = (all(values(laglengthfound)) ? 
      heuristicexpandfrac : 
      max(1.0, isfinite(heuristicexpandfrac) ? heuristicexpandfrac : 1.0))

    verbose && println("Convergence not indicated with $numsamplerecords samples. Est. convergence: ",
      round(1.0/(1+heuristicexpandfrac) * 100, RoundUp,digits=1),"%")
    for k ∈ keys(stopresults)
      (!stopresults[k]) && print("$k (ess=$(stopesss[k] |> round |> Int)/$miness), ")
    end


  end

  return (;stop, heuristicexpandfrac)
end
