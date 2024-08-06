

function derived(V::Val{:ypr}, dgp::AbstractDGP; kwargs...)
  @unpack μy, Σy = updatepy(dgp)
  ŷ = draw(Val{:y}(); μy, Σy)

  return seriesstats(V, dgp;est=ŷ, act=dgp.data.y)
end
  

function derived(V::Val{:yx}, dgp::AbstractDGP; testderived=false)
  @unpack dims, Θ = dgp
  @unpack x,ϕ = Θ
  @unpack s2t, Δt = dims
  X̃L = formX̃L(; x, dims)
  xS = formxS(x; dims)
  ŷ = X̃L * ϕ .+ xS

  testderived &&  @assert ŷ ≈ formΦ(ϕ;dims)*x

  return seriesstats(V, dgp;est=ŷ, act=dgp.data.y)
end

function derived(V::Val{:EyF}, dgp::AbstractDGP; testderived=false)
  @unpack dims, Θ, data =dgp
  @unpack ϕ,β = Θ
  @unpack s2t, Δt = dims
  @unpack F,r,y = data

  x̂ = F*β + r
  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)

  ŷ = X̃L * ϕ .+ xS

  testderived &&  @assert ŷ ≈ formΦ(ϕ;dims)*x̂

  return seriesstats(V, dgp;est=ŷ, act=y)
end

derived(V::Val{:yF_full},args...; includecumsumstat=false, kwargs...)=derived(
  Val{:yF}(), args...; includecumsumstat, seriesstattype=V, kwargs...)

derived(V::Val{:yF_full_cum},args...; kwargs...)=derived(
  Val{:yF_full}(), args...; includecumsumstat=true, kwargs...)
function derived(V::Val{:yF}, dgp::AbstractDGP; 
  data=dgp.data,
  dims=dgp.dims,
  testderived=false, 
  includecumsumstat=false,
  seriesstattype=V,
  kwargs...)
  @unpack Θ =dgp
  @unpack ϕ,β,ν, τy, τx = Θ
  @unpack T = dims
  @unpack F,r = data


  μx = F*β+r
  αψ= ν/2
  ζψ=ν/2
  ψ = draw(Val{:ψ}(); αψ, ζψ, T )


  x̂ = map(μx, ψ) do μxt, ψt
    rand(Normal(μxt, sqrt(inv(ψt*τx*τy))))
  end


  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)

  μy = X̃L * ϕ .+ xS
  ŷ = predictivedraw(dgp; F,r)

 
  stats = (typeof(data) <: Data ? 
    seriesstats(seriesstattype, dgp;est=ŷ, act=data.y,includecumsumstat) :  
    seriesstats(seriesstattype, dgp;est=ŷ, includecumsumstat))


  return stats
end


derived(V::Val{:yF_llike}, dgp::AbstractDGP; ) = (;yF_llike=lpdist(dgp,:yF_llike))
  


seriesstats(V::Val{Tstat}, args...; kwargs...) where Tstat = seriesstats(args...; label=Tstat, kwargs...)
function seriesstats(dgp::AbstractDGP; label::Symbol, est, act=nothing, includecumsumstat=false)
  @unpack Δt = dgp.dims
  meanest = mean(est)
  stats = Dict(
    label=>est,
    Symbol(:mean,label) => meanest,
    Symbol(:mean,label,:_scaled12)=>meanest |> r->r/Δt*12,
    Symbol(:mean,label,:_scaled12simp)=>meanest |> r->exp(r/Δt*12)-1,
    Symbol(:sum,label) => est |> sum,
    Symbol(:sum2, label,) => est.*est |> sum,
    Symbol(:σ2, label) => var(est),
  )

  if includecumsumstat
    stats[Symbol(label, :_cum)] = cumsum(est)
  end

  if act !== nothing
    stats[Symbol(:sum,label, :ε)] = sum(act-est)
    stats[Symbol(:sum2,label, :ε)] = sum((act-est).^2)
  end

  return stats

end

derived(V::Val{:xF_full},args...; includecumsumstat=false, kwargs...)=derived(
  Val{:xF}(), args...;  seriesstattype=V, includecumsumstat, kwargs...)

derived(V::Val{:xF_full_cum},args...; kwargs...)=derived(
    Val{:xF_full}(), args...; includecumsumstat=true, kwargs...)

function derived(V::Val{:xF}, dgp::AbstractDGP; 
  data=dgp.data,
  dims=dgp.dims,
  testderived=false, 
  seriesstattype=V,
  includecumsumstat = false)



  μx = F*β+r
  ψ = draw(Val{:ψ}(); αψ= ν/2, ζψ=ν/2, T )


  x̂ = map(μx, ψ) do μxt, ψt
    rand(Normal(μxt, sqrt(inv(ψt*τx*τy))))
  end
  return seriesstats(seriesstattype, dgp;est=x̂, includecumsumstat)
end

derived(::Val{:σ2x}, dgp; kwargs...) = (;σ2x= var(dgp.Θ.x))

#calculations for variances
function derived(::Val{:σ2_calc}, dgp;  ΣFr_full=nothing, )
  @unpack dims, Θ, data =dgp
  @unpack ϕ,β, τy,τx, ν, ϕ   = Θ
  @unpack s2t, Δt, T,S = dims
  @unpack F,r = data

  ϕ̃ = formϕ̃(; ϕ, dims)

  β1 = [β; 1.0]
  Fr = [F r]

  σ2x_calc = β1'cov(Fr)*β1+ ν/(ν-2)*1/(τx*τy)
  σ2x_calc_adj = σ2x_calc+1/τy/Δt

  x̂ = F*β + r
  @assert x̂ ≈ Fr*β1
  sumx_calc = sum(x̂)
  sum2x_calc = T*(σ2x_calc .+ (sum(x̂)/T)^2)
  sum2x_calc_adj = T*(σ2x_calc_adj .+ (sum(x̂)/T)^2)
  @assert σ2x_calc ≈ sum2x_calc/T - (sumx_calc/T)^2


  σ2y_calc = ϕ̃'ϕ̃*σ2x_calc + 1/τy
  X̃L = formX̃L(;x=x̂, dims)
  xS = formxS(x̂; dims) 
  sumy_calc = (X̃L*ϕ + xS) |> sum
  sum2y_calc = S*(σ2y_calc + (sumy_calc/S)^2)
  @assert σ2y_calc ≈ sum2y_calc/S - (sumy_calc/S)^2


  σ2xε_calc = ν/(ν-2)*1/(τx*τy)
  σ2xε_calc_adj = σ2xε_calc + 1/τy/Δt
  σ2yε_calc = ϕ̃'ϕ̃*σ2xε_calc + 1/τy
  σxε_calc_scaledsqrt12 = σ2xε_calc*12 |> sqrt
  σxε_calc_adj_scaledsqrt12 = σ2xε_calc_adj*12 |> sqrt

  lrparams = Dict()
  if ΣFr_full !== nothing
    lrparams[:σ2lrsys] = β1'ΣFr_full*β1
    lrparams[:σlrsys_scaledsqrt12] = σ2lrsys*12 |> sqrt
  end

  #σ2lrsysrfr = σ2lrsys + σ2rlr
  
  #σlrsysrfr_scaledsqrt12 = σ2lrsysrfr*12 |> sqrt

  return (; σ2x_calc, σ2x_calc_adj, sum2x_calc, sumx_calc, sum2x_calc_adj,
    σ2xε_calc, σ2xε_calc_adj,σxε_calc_scaledsqrt12, σxε_calc_adj_scaledsqrt12,
    σ2y_calc, sumy_calc, sum2y_calc, σ2yε_calc,  
    lrparams... )

end


function derived(::Val{:σ2_calc2}, dgp)
  @unpack dims, Θ, data =dgp
  @unpack ϕ,β, τy,τx, ψ, ϕ   = Θ
  @unpack s2t, Δt, T,S = dims
  @unpack F,r = data

  ϕ̃ = formϕ̃(; ϕ, dims)
  Φ = formΦ(ϕ; dims)


  σ2x_calc2 = [β; 1.0]'cov([F r])*[β; 1.0]+ mean(inv.(ψ*τx*τy))
  σ2x_calc2_adj = σ2x_calc2+1/τy/Δt

  sumx_calc2 = sum(F*β + r)
  sum2x_calc2 = T*(σ2x_calc2 .+ (sum(F*β)/T)^2)
  sum2x_calc2_adj = T*(σ2x_calc2_adj .+ (sum(F*β+r)/T)^2)
  @assert σ2x_calc2 ≈ sum2x_calc2/T - (sumx_calc2/T)^2


  σ2y_calc2 = ϕ̃'ϕ̃*σ2x_calc2 + 1/τy
  sumy_calc2 = Φ*F*β |> sum
  sum2y_calc2 = S*(σ2y_calc2 + (sumy_calc2/S)^2)
  @assert σ2y_calc2 ≈ sum2y_calc2/S - (sumy_calc2/S)^2


  σ2xε_calc2 = mean(inv.(τx*τy.*ψ))
  σ2xε_calc2_adj = σ2xε_calc2 + 1/τy/Δt
  σ2yε_calc2 = ϕ̃'ϕ̃*σ2xε_calc2 + 1/τy

  return (; σ2x_calc2, σ2x_calc2_adj, sum2x_calc2, sumx_calc2, sum2x_calc2_adj, 
    σ2xε_calc2, σ2xε_calc2_adj,
    σ2y_calc2, sumy_calc2, sum2y_calc2, σ2yε_calc2,  )

end




#loops through all derived parameters and supplies associated arguments
function addderived(;dgp, 
  predictionmethods=[PARAM[:iopredictionmethods]; PARAM[:iopredictionmethods_full]], 
  additionalderivedstats=PARAM[:ioadditionalderivedstats],
  modelfactors=PARAM[:livemodelfactors],
  Frf = PARAM[:factorFrf],
  factorfrequency=PARAM[:factorfrequency],
  tlocalparams=PARAM[:iotlocalparams],)

  @unpack dims=dgp
  @unpack K,P,Δt,Fθ2name=dims
  
  extraargsindex = Dict()
  if Symbol(":y[1]") ∈ keys(Fθ2name)    
    @unpack dims_full, data_full, dims_full_asif_mo, data_full_asif_mo = extendfactordata(
        dgp, ;modelfactors, Frf, factorfrequency)

    #WARNING- this may not align completely with the covariance that would be calculated
    #using data_full (formed subsequently) due to any misalignment with the reported frequency
    #however, it should be very close and identical for monthly data.
    ΣFr_full = [data_full_asif_mo.F data_full_asif_mo.r] |> cov


    #some of the derived stats

    extraargsindex[:σ2_calc] = (; ΣFr_full,)
    extraargsindex[:xF_full] = (;data=data_full_asif_mo, dims=dims_full_asif_mo)
    extraargsindex[:yF_full] = (;data=data_full, dims=dims_full)
    extraargsindex[:xF_full_cum] = (;data=data_full_asif_mo, dims=dims_full_asif_mo)
    extraargsindex[:yF_full_cum] = (;data=data_full, dims=dims_full)
    
    
    #run some sanity check
    oldFθ2name = dims.Fθ2name |> deepcopy
    addθnames!(dims; Fθ2name=dims_full.Fθ2name)
    addθnames!(dims; Fθ2name=dims_full_asif_mo.Fθ2name)

    #verify we didn't overwrite any useful values
    @assert all(keys(oldFθ2name) .|> k->oldFθ2name[k] ≡ dims.Fθ2name[k])

    #shouldn't be non-equivalent overlapping content
    commonkeys = intersect(keys(dims_full.Fθ2name), keys(dims_full_asif_mo.Fθ2name)) |> collect
    @assert all( commonkeys .|> k-> dims_full.Fθ2name[k] ≡ dims_full_asif_mo.Fθ2name[k]) "
      Possible conflict in the naming of parameters in Fθ2name for full vs full_asif_mo
      $(commonkeys[commonkeys .|> k-> dims_full.Fθ2name[k] !== dims_full_asif_mo.Fθ2name[k]])"

    (:yF_full ∈ Fderivedstats) && (:yF_full_cum ∈ Fderivedstats) && throw("
      :yF_full_cum automatically returns :yF_full- remove :yF_full from iopredictionmethods")
    (:xF_full ∈ Fderivedstats) && (:xF_full_cum ∈ Fderivedstats) && throw("
      :xF_full_cum automatically returns :xF_full- remove :xF_full from ioadditionalderivedstats")
  else
    @info "Skipping full stats- date labels not found for y in Fθ2name. Full and lr params will not be computed."
  end

  #avoid duplicate stats
  Fderivedstats = [predictionmethods; additionalderivedstats]
  @assert allunique(Fderivedstats)


  for Fstat ∈ [predictionmethods; additionalderivedstats]
    #@info Fstat

    if haskey(extraargsindex, Fstat)
      extraargs = extraargsindex[Fstat]
    else
      extraargs = (;)
    end

    dgp = addderived(Fstat; dgp, extraargs...)
  end

  return dgp
end

#this function computes derived stats from existing draws
#needs a refactor
function addderived(Fstat::Symbol;dgp::DGP{TΘ}, 
  deriveddependencies=PARAM[:ioderiveddependencies],
  kwargs...) where TΘ<:AbstractModelParameters

  @unpack dims,records,data,hyper = dgp
  @unpack S,T = dims
  @unpack numrecords, numchains, chainfieldindex, chainparts  = records

  template = derived(Val{Fstat}(), dgp; kwargs...)
  Nstat = OrderedDict(k=>length(template[k]) for k ∈ keys(template))

  dependencies::Vector{Symbol}=deriveddependencies[Fstat] ≡ :all ? collect(propertynames(dgp.Θ)) :  deriveddependencies[Fstat]

  #make sure we have captured everything we need
  if !(setdiff(dependencies, dgp.records.fieldstocapture) |> isempty)
    @warn "Dependencies $(setdiff(dependencies, dgp.records.fieldstocapture)) are necessary for derived stat
      $Fstat but some or all were not captured in records. Ignoring request."
    return dgp
  end

  irrelevant = setdiff(fieldnames(TΘ), dependencies)
  
  #the below is to verify that we update only the correct parameters
  dgpderived = DGP(dgp; Dict(irrelevant .=> nothing)..., strict=false)
  dependencytypes = Dict(p=>typeof(getproperty(dgp.Θ,p)) for p ∈ dependencies)

  extractsavedθ(::Type{<:Real},v) = v[1]
  extractsavedθ(::Type{Tdep},v) where Tdep<:AbstractArray  = v |> Tdep

  #capture the derived stats
  @eval Main dependencies=$dependencies
  #=θs = map(Iterators.product(1:numrecords, 1:numchains)) do (r, c)
    dgpderived = DGP(dgpderived; 
      Dict(p=>extractsavedθ(dependencytypes[p], chainparts[p][r,:,c]) for p ∈ dependencies)...)


    return derived(Val{Fstat}(), dgpderived; kwargs...)
  end |> collapse=#

  #push the derived stats into the records object
  #local augmented=dgp.records
  #print("Timing derived Fstat $Fstat consisting of $(length(keys(template))) stat groups")
  #@eval Main Nstat=$Nstat
  #@eval Main template=$template

  θs = integrateposterior(dgpderived;dgpθtypes=dependencytypes) do dgpi
    derived(Val{Fstat}(), dgpderived; kwargs...)
  end  |> collapse

  augmentations = OrderedDict()
  for k ∈ keys(template)
    if Nstat[k] == 1
      stats = reshape(θs[k],numrecords, 1, numchains)
    else
      stats = [θs[k][r,c][n] for r ∈ 1:numrecords, n ∈ 1:Nstat[k], c ∈ 1:numchains]
    end
    #stats = cat([reduce(vcat,θs[k][:,c] .|> transpose) for c in 1:numchains], dims=3)
    #@eval Main stats=$stats
    #@eval Main θs=$θs
    @assert size(stats) ≡ (numrecords, Nstat[k],numchains)
    @assert all(stats .!== missing)
    #print("timing creation of $Fstat : $k")
    augmentations[k] = stats
    ##=@time =#augmented = augmentchainrecords(augmented; augmentation=stats, Fgroup=k)
  end
  augmented = augmentchainrecords(dgp.records, augmentations)


  dgpnew = DGP(;Θ=dgp.Θ, data,  hyper, dims, records=augmented)
  return dgpnew
end

#integrates a function across parameters formed from the posterior distribution
function integrateposterior(fdgp::Tfdgp, dgp::AbstractDGP; 
    records=dgp.records,
    dgpθtypes = Dict(p=>typeof(getproperty(dgp.Θ,p)) for p ∈ propertynames(dgp.Θ)),
    ) where Tfdgp<:Function
  @unpack dims,Θ = dgp
  @unpack numrecords, numchains, chainfieldindex, chainparts  = records


    #this extracts the parameter values for conversion into a model parameters object
  #we add the ability to truncate x
  extractsavedθ(::Type{<:Real},v) = v[1]
  extractsavedθ(::Type{Tdep},v) where Tdep<:AbstractArray  = v |> Tdep
  extractsavedθ(::Val{Fθ}, v; ) where Fθ =extractsavedθ(dgpθtypes[Fθ], v)

  #the below are special procedureds for handling the case where the dgp is truncated
  #but the records are not
  extractsavedθ(::Val{:x},v; )=extractsavedθ(dgpθtypes[:x], v[1:dims.T])
  extractsavedθ(::Val{:ψ},v; )=extractsavedθ(dgpθtypes[:ψ], v[1:dims.T])

  θs2extract = keys(dgpθtypes)

  vals = map(Iterators.product(1:numrecords, 1:numchains)) do (r,c)
    dgpi = DGP(dgp; 
      Dict(p=>extractsavedθ(Val{p}(), chainparts[p][r,:,c]; ) for p ∈ θs2extract)...)
    
    fdgp(dgpi;)
    
  end

  return vals
end