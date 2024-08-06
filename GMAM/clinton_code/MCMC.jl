

#################initial values ofr mcmc)

initmcmc(s::Symbol, args...; kwargs...)=initmcmc(Val(s), args...; kwargs...)

#initialize the means with expectaitons of the priors as appropriate
function baseline_initmcmc(::Val{:priorexpectation}, ::Type{TΘ};dims,data,hyper, 
  minvar=eps(Float64)^0.5, additionalΘlist...) where TΘ<:AbstractModelParameters

  @unpack K, T, P, S, s2allts, s2t, Δt = dims
  @unpack ϕ0,v,κ0,δ0, αν0,ζν0,αx0,ζx0,αy0,ζy0,νmax,νmin, βΔ0 = hyper
  @unpack F,y=data

  

  #easy ones first
  ϕ = hyper.ϕ0 |> deepcopy
  ω = moment(Val{:Eω}(); κ=κ0,δ=δ0)
  #τx = moment(Val{:Eτx}(); αx=αx0,ζx=ζx0)
  #τy = moment(Val{:Eτy}(); αy=αy0,ζy=ζy0)

  #expectation is dependent on variable selection
  #γ = floor(ω*K) |> Int |> (Ktrue->[trues(Ktrue); falses(K-Ktrue)]) |> Vector{Bool}
  #start with the mean model
  γ = [true; falses(K-1) |> Vector{Bool}]
  β = hyper.β0 + inv(m.formD(γ,v)) * hyper.βΔ0

  #use a linear interpolation for the initial guess of x
  xinit = missings(Float64, T)
  xinit[1:P] .= 0.0
  for s ∈ 1:S
    xinit[(s2t[s]-Δt+1):(s2t[s])] .= y[s] / Δt
  end
  x = xinit .|> Float64

  #asserts that we filled x
  @assert sum(x) !== missing

  #get the initial precistion for y from the factor values of x and ϕ
  x̂ = F*β
  X̃L = formX̃L(; x=x̂ , dims)
  ỹ = formỹ(;y,x=x̂ ,dims)
  τy = 1/max(var(ỹ .- X̃L * ϕ), minvar)
  

  #τx is based on the residuals
  τx = 1/max(var(x .- F*β), minvar)*(TΘ <: DGPModelParametersGT ? 1.0 : 1/τy)

  
  #intial value of ν is from the expectation of the untruncated version of the prior
  ν = αν0/ζν0
  
  #a reasonable value for ψ is tricky- compute as an expectation implied by previous values (see updating equations)
  αψ = fill(ν/2 + 0.5,T)
  ζψ = τx*τy/2 .* (x-F*β).^2 .+ ν/2
  ψ = αψ ./ ζψ

  baselineΘlist = (;x, ϕ, β, γ, ω, τy, τx, ψ, ν)
  Θlist = merge(baselineΘlist, additionalΘlist)
  #need to walk through part of the updating procedure to get the expectations for ν
  #=η1 = sum(log.(ψ) .- ψ) - 2ζν0
  lpν(ν) = (T*ν/2 + αν0-1)*log(ν/2) - T*loggamma(ν/2) + ν/2 * η1
  pν(ν) = lpν(ν) |> exp 
  ν = moment(Val{:Eν}();pν, νmax, νmin)=#

  Θ = TΘ(;Θlist...)
  #@eval Main t=$Θ
  return Θ
end


#fallback
initmcmc(initmethod::Val, args...;kwargs...)=baseline_initmcmc(initmethod, args...; kwargs...)

#g specific variant for the G model
function initmcmc(initmethod::Val, TΘ::Type{<:AbstractModelParametersG}; dims, data, hyper)

  #use the inverse expectation of the gamma since usually the inverse here will be used in the distributions
  τβ = hyper.αβ0/hyper.ζβ0
  τϕ = hyper.αϕ0/hyper.ζϕ0

  #@eval Main h=$hyper
  #throw("stop")
  
  return baseline_initmcmc(initmethod, TΘ; dims, data, hyper, τβ, τϕ)
end


#draws a parameter from the conditional distribution
#for each parameter, 1) get the conditional distributional parameters
#given all other parameters 2) make the draw and 3) update the current DGP object
function conditionaldraw(θ::Symbol, dgp,)


  if θ ≡ :ϕ
    @unpack ϕmin, ϕmax = dgp.hyper
    @unpack μϕ, Σϕ, Λϕ = updateθpϕ(dgp)
    ϕ = draw(:ϕ; μϕ,Σϕ,Λϕ,ϕmin,ϕmax)
    return DGP(dgp; ϕ)

  elseif θ ≡ :x
    @unpack μx, Σx = updateθpx(dgp)
    x = draw(:x; μx, Σx)
    return DGP(dgp; x)

  elseif θ ≡ :τy
    @unpack αy, ζy = updateθpτy(dgp)
    τy = draw(:τy; αy, ζy)
    return DGP(dgp; τy)

  elseif θ ≡ :τx
    @unpack αx, ζx = updateθpτx(dgp)
    τx = draw(:τx; αx, ζx)
    return DGP(dgp; τx)

  elseif θ ≡ :τϕ
    @unpack αϕ, ζϕ = updateθpτϕ(dgp)
    τϕ = draw(:τϕ; αϕ, ζϕ)
    return DGP(dgp; τϕ)  

  elseif θ ≡ :τβ
    @unpack αβ, ζβ = updateθpτβ(dgp)
    τβ = draw(:τβ; αβ, ζβ)
    return DGP(dgp; τβ)  

  elseif θ ≡ :β
    @unpack μβ, Σβ = updateθpβ(dgp)
    β = draw(:β; μβ, Σβ)  
    return DGP(dgp; β)

  #a bit more complex as the diagonal and the general scenarios are substantively different
  #=elseif θ ∈ (:γ_det, :γ_rand) && typeof(dgp.hyper.A0)<:Diagonal
      @unpack pγ = updateθpγ(dgp)
      γ = draw(:γ; pγ)
      return DGP(dgp; γ)=#
  elseif θ ∈ (:γ_det, :γ_rand)# && typeof(dgp.hyper.A0)<:Matrix
      dgpnew = DGP(dgp; γ=deepcopy(dgp.Θ.γ)) #will update this in-place, so form the new dgp early
      ks = collect(1:dgp.dims.K)
      if θ ≡ :γ_rand
        shuffle!(ks)
      else
        @assert θ ≡ :γ_det
      end

      for k ∈ ks
        @unpack pγ = updateθpγ(dgpnew, k, dgpnew.hyper.A0)
        dgpnew.Θ.γ[k] = draw(:γ, pγ)
      end
      #print("\n")
    return dgpnew
  elseif θ ≡ :ω
    @unpack κ, δ = updateθpω(dgp)
    ω = draw(:ω; κ, δ)
    return DGP(dgp; ω)

  elseif θ ≡ :ψ
    @unpack ζψ, αψ = updateθpψ(dgp)
    ψ = draw(:ψ; ζψ, αψ) 
    return DGP(dgp; ψ)
    
  #the below is a bit more complex due to the history required for Metropolis-Hastings
  elseif θ ≡ :ν
    @unpack rνdist, lrν = dgp.hyper
    νtM1 = dgp.Θ.ν

    lpν = updateθpν(dgp).lpν
    ν = draw(:ν; rνdist, lrν, νtM1, lpν)
    return DGP(dgp; ν)

  
  #not technically an update, at least until we start handling missing data
  elseif θ ≡ :y
    throw("Cannot draw y as a parameter. Missing values not (yet) supported!")
  else
    throw("unrecognized value of θ \'$θ\' for conditional draw")
  end

  @assert false
end





function mcmc(tapedeck::Ttapedeck; dgp::DGP, 
    stoprule,
    maxrecords,
    samplerecyclefrac=PARAM[:samplerecyclefrac],
    sampleexpandmethod=PARAM[:sampleexpandmethod],
    verbose=true
    ) where Ttapedeck<:Function

  @unpack dims, hyper,  = dgp
  @unpack numchains = dgp.records



  #create a DGP for each chain
  dgps = [DGP(dgp; copyΘ=true) for c ∈ 1:numchains]



  #we will stop this via the stopping rule at the end (Julia has no do-while loop)
  while true
    @unpack records=dgp
    @unpack numrecords,numsamplerecords,numburnrecords = records

    @assert numrecords ≤ maxrecords

    @assert allequal(records.lastrecord)
    lastrecord=records.lastrecord[1]
    @assert  lastrecord ≤ numrecords


    p=Progress((numrecords-lastrecord)*numchains,dt=1, showspeed=true)
    for r ∈ (lastrecord+1):numrecords
      for c ∈ 1:numchains      
        for θ ∈ tapedeck()
          dgps[c] = conditionaldraw(θ, dgps[c])
        end
        record!(dgps[c], chainnum=c)
        next!(p) #note- apparently this is thread-safe
      end
    end
    
    println()
    #now set the termination conditions
    checkresults = checkstoprule(stoprule; dgp, verbose, )
    checkresults.stop && break
    if numrecords == maxrecords
      @warn "Max iterations $maxrecords reached. 
        MCMC failed to converge according to stopping rule $stoprule rule. 
        Terminating chain- hope for the best 🤞"
      break
    end
    

    #now need to expand the records object
  
    if sampleexpandmethod ≡ :fixed
      sampleexpandfrac = PARAM[:samplefixedexpandfrac]
    elseif sampleexpandmethod ≡ :heuristic
      sampleexpandfrac = checkresults.heuristicexpandfrac
    else
      throw("Unrecognized sampleexpandmethod $sampleexpandmethod")
    end
    expanded =expandchainrecords(; 
      dgp, 
      samplerecyclefrac, 
      sampleexpandfrac,
      maxrecords)
    verbose && (@info "Convergence not reached, expanding records. 
      numsamplesize increased from $(numsamplerecords) to $(expanded.numsamplerecords); 
      numrecords increased from $numrecords to $(expanded.numrecords) (max=$maxrecords)")
    dgp = DGP(dgp; records=expanded)
    dgps = [DGP(dgps[c]; records=expanded) for c ∈ 1:numchains]
    
  end

  rescale!(dgp)

  return dgp
end

rescale!(::AbstractDGP) = nothing
function rescale!(dgp::AbstractDGP{<:DGPModelParametersGIR})
  @unpack dims, records = dgp
  @unpack T = dims

  records.β .*= 1/sqrt(T)

  return nothing
end

gettapedeck(mcmcruntype::Symbol, dgp::DGP) = gettapedeck(mcmcruntype|>Val, dgp)
gettapedeck(::Val{:deterministic1}, ::DGP{<:DGPModelParameters}) = ()->[:ω,:γ_det,:ν,:ψ,:β,:x,:ϕ,:τy,:τx,]
gettapedeck(::Val{:shuffle1}, ::DGP{<:DGPModelParameters}) = ()->shuffle([:ω,:γ_rand,:ν,:ψ,:β,:x,:ϕ,:τy,:τx,])

gettapedeck(::Val{:deterministic1}, ::DGP{<:AbstractModelParametersG}) = ()->[:ω,:τϕ,:τβ, :γ_det,:ν,:ψ,:β,:x,:ϕ,:τy,:τx,]
gettapedeck(::Val{:shuffle1}, ::DGP{<:AbstractModelParametersG}) = ()->shuffle([:ω,:τϕ,:τβ,:γ_rand,:ν,:ψ,:β,:x,:ϕ,:τy,:τx,])


mcmc(mcmcruntype, dgp::DGP; kwargs...) = mcmc(gettapedeck(mcmcruntype, dgp); dgp, kwargs...)



function profilemcmc(mcmcruntype, dgp::DGP{TModelParameters}; testvalues=:all) where TModelParameters
  tapedeck = gettapedeck(mcmcruntype, dgp)

  @info "*********************************************************************************"
  @info "*********************************************************************************"
  @info "*********************************************************************************"

  @info "Profiling model parameters type $TModelParameters with mcmc run type $mcmcruntype"

  #reset the seed
  Random.seed!(11)
  for θ ∈ tapedeck()
    (testvalues!==:all) && (θ ∉ testvalues) && continue
    @info "Now profiling $θ" 
    b = @benchmark conditionaldraw($θ,$dgp)
    display(b)
  end



  if (testvalues ≡ :all) || :record ∈ testvalues
    @info "Profiling record!"
    b=@benchmark begin
      function placeborecord(dgp)
      
        record!(dgp, chainnum=1)
        dgp.records.lastrecord[1] -= 1
      end
      placeborecord($dgp)
    end
    display(b)
  end

  @info "Profiling completed successfully."
end
