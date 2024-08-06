

loadprior(x, args...; kwargs...)=x
loadprior(s::Symbol,args...; kwargs...) = loadprior(Val(s), args...; kwargs...)

loadprior(::Val{:phi0_uniform}, ::Any; dims, kwargs...) = [1/(dims.P+Δt) * dims.Δt for p ∈ 1:dims.P]
loadprior(::Val{:phi0_contemp}, ::Any; dims, kwargs...) = zeros(dims.P)

#the below needs to be refactored and integrated into the parameter sheet, with a custom β name dictionary

loadprior(::Val{:beta0_zeros}, ::Any; dims, kwargs...) = zeros(dims.K)
#loadprior(::Val{:A0_oil14}; dims, kwargs...) = [0.01; fill(0.001,2); 10^-2; fill(0.001,10)] |> Diagonal


#loadprior(::Val{:M0_homosked1}, ::Type{<:DGPModelParameters}; dims, kwargs...) = 0.01 .* ones(dims.P) |>Diagonal

loadprior(::Val{:A0_froma0}, ::Type{<:AbstractModelParametersG}; dims, kwargs...) = ones(dims.K) |>Diagonal

loadprior(::Val{:zetaphi0_zellnersiow}, ::Any; dims, kwargs...)  = dims.S/2
loadprior(::Val{:zetabeta0_zellnersiow}, ::Any; dims, kwargs...)  = dims.T/2

loadprior(::Val{:zetabeta0_cauchy}, ::Any; dims, kwargs...)  = 1/2
loadprior(::Val{:zetaphi0_cauchy}, ::Any; dims, kwargs...)  = 1/2

#for the below- see Gupta and Ibrahim 2009 for justification of a convex combo here
loadprior(::Val{:A0_halfzellner}, ::Any; dims, data, F=data.F, kwargs...) = F'F .* 0.5 + Diagonal(F'F) .* 0.5
loadprior(::Val{:beta0_fromsim}, ::Any; dims, kwargs...) =  fill(PARAM[:simulatebeta0], dims.K)
loadprior(::Val{:betadelta0_fromsim}, ::Any; dims, kwargs...) =  fill(PARAM[:simulatebetadelta0], dims.K)
#loadprior(V::Any,args...; kwargs...) = throw("loadprior failed for prior type $V. kwargs: $kwargs")

function loadprior(::Val{:beta0_zeroswithoverrides}, ::Any; dims, d, overrides = d[:beta0overrides], kwargs...)
  @unpack K,Fθ2name = dims
  βsyms = [Symbol(:β,"[$i]") for i ∈ 1:K]
  #β=zeros(K)

  numoverrides=0
  β = broadcast(1:K) do k
    Fβk = Fθ2name[βsyms[k]]
    if haskey(overrides, Fβk)
      numoverrides += 1
      return overrides[Fβk]
    else
      return 0.0
    end
  end

  @assert numoverrides == length(keys(overrides))

  return β
end


#this version adjusts the total precision prior of β to 1 (excluding the effect of γ)
function loadprior(::Val{:A0_adjdefaultwithoverrides}, args...; d, a0default=d[:a0default], 
    kwargs...,)
    αy0 =loadprior(d[:alphay0], args...; d, kwargs...)
    ζy0 =loadprior(d[:zetay0], args...; d, kwargs...)

    αx0 =loadprior(d[:alphax0], args...; d, kwargs...)
    ζx0 =loadprior(d[:zetax0], args...; d, kwargs...)

    αβ0 =loadprior(d[:alphabeta0], args...; d, kwargs...)
    ζβ0 =loadprior(d[:zetabeta0], args...; d, kwargs...) 



  adja0default = inv(αy0/ζy0)*inv(αx0/ζx0)*inv(αβ0/ζβ0) .* a0default

  return loadprior(Val{:A0_defaultwithoverrides}(), args...;  d, a0default=adja0default,kwargs...)
end

function loadprior(::Val{:A0_defaultwithoverrides}, ::Any;  
  dims, d, overrides = d[:a0overrides], a0default=d[:a0default], kwargs...)
  @unpack K,Fθ2name = dims

  #coefficient names correspond to that of beta
  βsyms = [Symbol(:β,"[$i]") for i ∈ 1:K]
  #a0=[a0default for k ∈ 1:K]

  numoverrides=0
  a0 = broadcast(1:K, a0default) do k, a0defaultk
    Fa0k = Fθ2name[βsyms[k]]
    if haskey(overrides, Fa0k)
      numoverrides += 1
      return overrides[Fa0k]*a0defaultk
    else
      return a0defaultk
    end
  end

  @assert numoverrides == length(keys(overrides))

  return a0 |> Diagonal
end

#this version adjusts the total precision prior of ϕ to 1.0
function loadprior(::Val{:M0_adjdefaultwithoverrides}, args...; d, m0default=d[:m0default], 
  kwargs...,
  ) 

  αy0 =loadprior(d[:alphay0], args...; d, kwargs...)
  ζy0 =loadprior(d[:zetay0], args...; d, kwargs...)
  αϕ0 =loadprior(d[:alphaphi0], args...; d, kwargs...)
  ζϕ0 =loadprior(d[:zetaphi0], args...; d, kwargs...) 

  adjm0default = inv(αy0/ζy0)*inv(αϕ0/ζϕ0) .* m0default

  return loadprior(Val{:M0_defaultwithoverrides}(), args...;  d, m0default=adjm0default, kwargs...)
end


function loadprior(::Val{:M0_defaultwithoverrides}, ::Any; 
  dims, d, overrides = d[:m0overrides], m0default=d[:m0default], kwargs...)
  
  @unpack P = dims

  #coefficient names correspond to that of beta
  m0syms = [Symbol(:m0,"[$p]") for p ∈ 1:P]
  #m0=[m0default for p ∈ 1:P]


  numoverrides=0
  m0 = broadcast(1:P, m0default) do p, m0defaultp
    if haskey(overrides, m0syms[p])
      numoverrides += 1
      return overrides[m0syms[p]] *  m0defaultp
    else
      return m0defaultp
    end
  end

  @assert numoverrides == length(keys(overrides))

  return m0 |> Diagonal
end


#create the baseline hyperparameters
function baseline_loadpriorsfromasciidict(d,TΘ::Type{<:AbstractModelParameters}; dims, data, additionalhyper...)

  @unpack P, K, T = dims
  @unpack F,y,r = data

  #short-hand version of loadparior
  lp(k) = loadprior(d[k], TΘ; dims, data, d)

  #hyperparameters- some of these are set in advance due to dependencies
  ζν0=lp(:zetanu0) #corresponds to mode 4, variance 200
  αν0=lp(:alphanu0) 
  rνdist=truncated(Gamma(αν0, 1/ζν0),lp(:numin), lp(:numax)) #set proposal to prior
  lrν=(x)-> logpdf(rνdist,x)

  hypert = (;
    x = (;),  
    ϕ = (;
      ϕ0 = lp(:phi0),
      M0 = lp(:M0), 
      ϕmin = lp(:phimin),
      ϕmax = lp(:phimax),
      ),  
    β = (;
      v=lp(:v),
      β0 = scaleβprior(lp(:beta0), TΘ; dims),
      βΔ0 = scaleβprior(lp(:betadelta0), TΘ; dims),
      A0 = lp(:A0; )),
    γ = (;),
    ω = (;
      κ0=lp(:kappa0),
      δ0=lp(:delta0)),    
    τy = (;
      αy0=lp(:alphay0),
      ζy0=lp(:zetay0)),      
    τx = (;
      αx0=lp(:alphax0),
      ζx0=lp(:zetax0)),     
    ψ = (;),
    ν = (;
      νmax=lp(:numax),
      νmin=lp(:numin),
      ζν0,
      αν0, 
      rνdist, #set proposal to prior
      lrν,)
    )
  @assert allunique([collect(hypert |> keys); collect(additionalhyper |> keys)])

  hypert = merge(hypert, additionalhyper)
  hyper = TΘ(; hypert...)

  @eval Main hyper=$hyper

  return hyper
end

scaleβprior(βprior, ::Type{<:AbstractModelParameters}; dims) = βprior
scaleβprior(βprior, ::Type{<:DGPModelParametersGIR}; dims) = βprior .* sqrt(dims.T)
scaleζxprior(ζ, ::Type{<:DGPModelParametersGIR}; dims) = βprior .* sqrt(dims.T)




#fallback
loadpriorsfromasciidict(args...;kwargs...)=baseline_loadpriorsfromasciidict(args...; kwargs...)

#a specific variant for the G model
function loadpriorsfromasciidict(d, TΘ::Type{<:AbstractModelParametersG}; dims, data, )

  τϕ= (; 
    αϕ0 = loadprior(d[:alphaphi0], TΘ; dims, data),
    ζϕ0 = loadprior(d[:zetaphi0], TΘ; dims, data))
  τβ= (;
    αβ0 = loadprior(d[:alphabeta0], TΘ; dims, data),
    ζβ0 = loadprior(d[:zetabeta0], TΘ; dims, data)
  )


  
  return baseline_loadpriorsfromasciidict(d, TΘ; dims, data, τβ,τϕ)
end
#loads a prior set from an already completed analsysi
function loadpriorsfrompreviousresults(priorparam, 
    TΘ::Type{<:Union{DGPModelParametersG,DGPModelParametersGIR}};
    rid,
    priorsetrid = priorparam[:estimatepriormodels] ? rid : priorparam[:rid],
    analysispath=PARAM[:analysispath],
    fundname,
    dims,  
    data,
    priorlinkindex=priorparam[:priorlinkindex],
    betapriormethod=priorparam[:betapriormethod],
    impliedm0asparam = priorparam[:impliedm0asparam], 
    implieda0asparam = priorparam[:implieda0asparam], 
    βpriorparam = priorparam[:betapriorparam],
    βpriorprecmax = priorparam[:betapriorprecmax],
    βpriorprecmin = priorparam[:betapriorprecmin],
    ϕpriorprecmax = priorparam[:phipriorprecmax],
    ϕpriorprecmin = priorparam[:phipriorprecmin],
    priorpriorset=priorparam[:priorpriorset],
    ϕ0min = priorparam[:phi0min],
    ϕ0max = priorparam[:phi0max],
    skiphistpriors=priorparam[:skiphistpriors],
    zerointerceptprior=priorparam[:zerointerceptprior],
    minconditionalsamplesize=priorparam[:minconditionalsamplesize],
    minconditionaldeviation=priorparam[:minconditionaldeviation],
    testhistpriors=true
    )


  @unpack P,S,T,K, Δt = dims


  hyperdef = loadpriors(priorpriorset, TΘ; dims, data, )
  if fundname ∉ keys(priorlinkindex)
    @assert fundname ∈ values(priorlinkindex)
    return hyperdef
  end

  priorfundname = priorlinkindex[fundname]

  summaryfiles = readdir("$analysispath/$priorsetrid/summary/whole")
  filter!(f->match(Regex("[a-zA-Z0-9_]*$(priorfundname)_[a-zA-Z0-9_]*$priorsetrid.csv"),f) !== nothing, summaryfiles)
  if length(summaryfiles)>1
    throw("Multiple summary files found for fund for priorfundname $priorfundname
      foundfiles: $summaryfiles")
  elseif length(summaryfiles) < 1
    return nothing
  end

  d = CSV.File("$analysispath/$rid/summary/whole/$(summaryfiles[begin])") |> DataFrame
  d.chain .= d.chain .|> Symbol
  chainname = ifelse(length(d.chain |> unique) == 1 , d.chain[begin], :full)
  d = d[d.chain .≡ chainname,:]

  N = d.N[begin]
  v = d.v[begin]

  d.E = d.E .|> x->x≡missing ? missing : x
  d.median = d.med .|> x->x≡missing ? missing : x
  d.paramgroup = d.paramgroup .|> Symbol
  d.param = d.param .|> Symbol

  dparam = groupby(d,:param)
  @assert all(combine(dparam,nrow).nrow .== 1)
  param2Eidx = Dict(Fθ=>dparam[(;param=Fθ)][begin,:E] for Fθ ∈ d.param)
  param2noteidx = Dict(Fθ=>dparam[(;param=Fθ)][begin,:note] for Fθ ∈ d.param)
  dparamg = groupby(d, :paramgroup)
  @assert length(dparam) == nrow(d)



  function momgamma(;μ,σ2)
    α = μ^2/σ2
    ζ = μ/σ2
    @assert mean(Gamma(α,1/ζ)) ≈ μ
    @assert var(Gamma(α,1/ζ)) ≈ σ2
    return (α,ζ)
  end
  v::Float64 = d.v[begin]

  #priors for τ
  αy0::Float64, ζy0::Float64 = param2Eidx[:p_αy0], param2Eidx[:p_ζy0]
  αx0::Float64, ζx0::Float64 = param2Eidx[:p_αx0], param2Eidx[:p_ζx0]
  αβ0::Float64, ζβ0::Float64 = param2Eidx[:p_αβ0], param2Eidx[:p_ζβ0]
  αϕ0::Float64, ζϕ0::Float64 = param2Eidx[:p_αϕ0], param2Eidx[:p_ζϕ0]


  #priors for β- two scenarios
  if βpriorparam ≡ :β0
    β0::Vector{Float64} = map(1:K) do k
      param2Eidx[Symbol("p_β0[$k]")]
    end
    βΔ0::Vector{Float64} = zeros(K)
    β0[1] *= (!zerointerceptprior)
    #in this case, we do not want the selection biased towards 0.5
    #κ0::Float64, δ0::Float64 = (1.0,1.0)
  elseif βpriorparam ≡ :βΔ0
    β0 = map(1:K) do k
      eval(Meta.parse(dparam[(;param=Symbol(:β,"[$k]"))][begin,:prior])).β0 |> Float64
    end
    βΔ0 = map(1:K) do k
      param2Eidx[Symbol("p_βΔ0[$k]")]
    end   
    βΔ0[1] *= (!zerointerceptprior)
   
  #this version uses the conditional distribution to pick β0 after performing some snaity checks
  elseif βpriorparam ≡ :cγ_β0
    param2Nidx = Dict(Fθ=>dparam[(;param=Fθ)][begin,:N] for Fθ ∈ d.param)

    β0 = map(1:K) do k
      β0prev = eval(Meta.parse(dparam[(;param=Symbol(:β,"[$k]"))][begin,:prior])).β0 |> Float64
      β0cand = param2Eidx[Symbol(:cγ_p_β0,"[$k]")]
      Nk = param2Nidx[Symbol(:cγ_p_β0,"[$k]")]
      if (Nk > minconditionalsamplesize) && abs(β0cand-β0prev) > minconditionaldeviation 
        return β0cand
      else 
        return β0prev
      end
    end
    βΔ0 = zeros(K)
    β0[1] *= (!zerointerceptprior)  
  else
    @assert false
  end
 
  @assert (param2noteidx[Symbol("β[1]")] |> string) == "intercept"
  

  #prior for ω
  κ0, δ0 = param2Eidx[:p_κ0], param2Eidx[:p_δ0]
  #κ0, δ0 = 0.5, param2Eidx[:p_δ0]

  if implieda0asparam
    a0::Vector{Float64} = map(1:K) do k
      a0kraw = param2Eidx[Symbol("p_a0[$k]")]
      impliedτ = a0kraw*αy0/ζy0*αx0/ζx0*αβ0/ζβ0
      a0k = a0kraw*max(min(impliedτ, βpriorprecmax),βpriorprecmin)/impliedτ
      return a0k
    end

    @assert all(βpriorprecmin .⪅ (a0.*αy0/ζy0*αx0/ζx0*αβ0/ζβ0) .⪅ βpriorprecmax) " 
      βpriorprecmin=$βpriorprecmin  and βpriorprecmax=$βpriorprecmax but
        a0*αy0/ζy0*αx0/ζx0*αβ0/ζβ0=$(a0*αy0/ζy0*αx0/ζx0*αβ0/ζβ0) !"
  else
    a0 = map(1:K) do k
      eval(Meta.parse(dparam[(;param=Symbol(:β,"[$k]"))][begin,:prior])).A0 |> Float64
    end
    @assert all(a0 .≈ diag(hyperdef.A0))

  end
  A0 = Diagonal(a0)


  #priors for ϕ
  ϕmin::Float64 = priorparam[:phimin]
  ϕmax::Float64 = priorparam[:phimax]
  ϕ0::Vector{Float64} = map(1:P) do p
    min(max(param2Eidx[Symbol("p_ϕ0[$p]")],ϕ0min),ϕ0max)
  end
  if impliedm0asparam
    m0::Vector{Float64} = map(1:P) do p
      m0praw = param2Eidx[Symbol("p_m0[$p]")]
      impliedτ = m0praw*αy0/ζy0*αϕ0/ζϕ0
      m0p = m0praw*max(min(impliedτ, ϕpriorprecmax),ϕpriorprecmin)/impliedτ
      return m0p
    end
    @assert all(ϕpriorprecmin .⪅ m0*αy0/ζy0*αϕ0/ζϕ0 .⪅ ϕpriorprecmax)
  else
    m0= map(1:P) do p
      eval(Meta.parse(dparam[(;param=Symbol(:ϕ,"[$p]"))][begin,:prior])).M0 |> Float64
    end
    @assert all(m0 .≈ diag(hyperdef.M0))
  end
  M0=Diagonal(m0)

  #priors for ν
  νmin::Float64 = priorparam[:numin]
  νmax::Float64 = priorparam[:numax]
  @assert νmax ≡ Inf "νmax must = ∞ for histpriors"
  νμ::Float64 = max(dparam[(;param=:ν)][begin, :E], νmin+1)
  νσ2::Float64 = dparam[(;param=:ν)][begin, :sd]^2
  
  local αν0::Float64, ζν0::Float64
  try
    matchedmoments = matchtruncatedgammamoments(;lower=νmin, μ= νμ, σ2=νσ2,)
    matchedmoments.flag || error("Convergence flag = $(matchedmoments.flag)")
    αν0, ζν0 = matchedmoments.α0,matchedmoments.ζ0
  catch err
    @info "WARNING: Failed to converge on hyperparameters for αν0, ζν0 with error $err.
      using moments impleid by untruncated distribution as an approximation."
      αν0, ζν0 = momgamma(;μ=νμ,σ2=νσ2,)
  end

  rνdist=truncated(Gamma(αν0, 1.0/ζν0), νmin, νmax)  #set proposal to prior
  lrν=(x)-> logpdf(rνdist,x)




  hypert = (;
    ϕ = (;ϕ0, M0, ϕmin, ϕmax),
    β = (;v, β0, βΔ0, A0),
    τy = (;αy0, ζy0),
    τx = (;αx0, ζx0),
    τβ = (;αβ0, ζβ0),
    τϕ = (;αϕ0, ζϕ0),
    ν = (;αν0, ζν0, rνdist, lrν, νmin, νmax),
    ω = (; κ0, δ0,),
    x = (;),
    ψ = (;),
    γ = (;),
  )

  #override the histpriors with the default- helpful for testing
  for Fθ ∈ skiphistpriors

    θdef = getproperty(hyperdef,Fθ)

    @assert haskey(hypert, Fθ) || any([haskey(hypert[k],Fθ) for k ∈ keys(hypert)])
    #swap all priors associated with a particular parameter for the default
    if haskey(hypert, Fθ)
      hypert = merge(hypert, Dict(Fθ=>θdef))
      continue
    end
    
    merged = false
    #swap a single prior associated with a particular parameter for the default
    for k ∈ keys(hypert)
      if haskey(hypert[k],Fθ)
        hypert = merge(hypert, Dict(k=>merge(hypert[k],Dict(Fθ=>θdef))))
        merged=true
        break
      end
    end
    @assert merged
  end


  hyper = TΘ(; hypert...)

  return hyper
end

#loads a prior set from an already completed analsysi
function loadpriorsfrompreviousresults_old(priorparam, 
    TΘ::Type{<:Union{DGPModelParametersG,DGPModelParametersGIR}};
    rid,
    priorsetrid = priorparam[:estimatepriormodels] ? rid : priorparam[:rid],
    analysispath=PARAM[:analysispath],
    fundname,
    dims,  
    data,
    priorlinkindex=priorparam[:priorlinkindex],
    betapriormethod=priorparam[:betapriormethod],
    pγmin = priorparam[:betapgammamin],
    pγmax = priorparam[:betapgammamax],
    a0max = priorparam[:betaa0max],
    a0min = priorparam[:betaa0min],
    priorpriorset=priorparam[:priorpriorset],
    ϕ0min = priorparam[:phi0min],
    ϕ0max = priorparam[:phi0max],
    skiphistpriors=priorparam[:skiphistpriors],
    testhistpriors=true
    )


  @unpack P,S,T,K, Δt = dims


  hyperdef = loadpriors(priorpriorset, TΘ; dims, data, )
  if fundname ∉ keys(priorlinkindex)
    @assert fundname ∈ values(priorlinkindex)
    return hyperdef
  end

  priorfundname = priorlinkindex[fundname]

  summaryfiles = readdir("$analysispath/$priorsetrid/summary/whole")
  filter!(f->match(Regex("[a-zA-Z0-9_]*$(priorfundname)_[a-zA-Z0-9_]*$priorsetrid.csv"),f) !== nothing, summaryfiles)
  if length(summaryfiles)>1
    throw("Multiple summary files found for fund for priorfundname $priorfundname
      foundfiles: $summaryfiles")
  elseif length(summaryfiles) < 1
    return nothing
  end

  d = CSV.File("$analysispath/$rid/summary/whole/$(summaryfiles[begin])") |> DataFrame
  d.chain .= d.chain .|> Symbol
  chainname = ifelse(length(d.chain |> unique) == 1 , d.chain[begin], :full)
  d = d[d.chain .≡ chainname,:]

  N = d.N[begin]
  v = d.v[begin]

  d.E = d.E .|> Float64
  d.median = d.med .|> Float64
  d.paramgroup = d.paramgroup .|> Symbol
  d.param = d.param .|> Symbol

  dparam = groupby(d,:param)
  dparamg = groupby(d, :paramgroup)
  @assert length(dparam) == nrow(d)

  #priors for τ
  αy0, ζy0 = momgamma(;μ=dparam[(;param=:τy)][begin, :E], σ2=dparam[(;param=:τy)][begin, :sd]^2)
  αx0, ζx0 = momgamma(;μ=dparam[(;param=:τx)][begin, :E], σ2=dparam[(;param=:τx)][begin, :sd]^2)
  αβ0, ζβ0 = momgamma(;μ=dparam[(;param=:τβ)][begin, :E], σ2=dparam[(;param=:τβ)][begin, :sd]^2)
  αϕ0, ζϕ0 = momgamma(;μ=dparam[(;param=:τϕ)][begin, :E], σ2=dparam[(;param=:τϕ)][begin, :sd]^2)

  #prior for ϕ
  @assert nrow(dparamg[(;paramgroup=:ϕ)]) == P
  ϕ0 = map(1:P) do p
    max(min(dparam[(;param=Symbol(:ϕ,"[$p]"))][begin, :E],ϕ0max),ϕ0min)
  end
  m0 = map(1:P) do p
    1/dparam[(;param=Symbol(:ϕ,"[$p]"))][begin, :sd]^2*inv(αy0/ζy0)*inv(αϕ0/ζϕ0) 
  end
  ϕmin = priorparam[:phimin]
  ϕmax = priorparam[:phimax]
  M0 = Diagonal(m0)

  #priors for ν
  #the below isn't exactly right due to the truncation
  αν0, ζν0 = momgamma(;μ=dparam[(;param=:ν)][begin, :E], σ2=dparam[(;param=:ν)][begin, :sd]^2)
  νmin = priorparam[:numin]
  νmax = priorparam[:numax]
  rνdist=truncated(Gamma(αν0, 1.0/ζν0), νmin, νmax)  #set proposal to prior
  lrν=(x)-> logpdf(rνdist,x)


  #β is a little more complex due to the Bernoulli-mixture nature of the variable
  #see the model doc appendix
  #match the expected precisions
  function βprior(::Val{:zerobetadelta0}; p,v,varβ, Eβ, kwargs...) 
    β0=Eβ
    βΔ0=0.0
    #a0raw=max(a0min, min(a0max, (p+(1-p)*v^2)/varβ))
    varβc= 1/max(a0min, min(a0max, 1/varβ))
    #a0 = max(a0min, min(a0max, (p+(1-p)*v^2)/varβc)) .* inv(αy0/ζy0)*inv(αx0/ζx0)*inv(αβ0/ζβ0)
    a0 = 1/varβc * inv(αy0/ζy0)*inv(αx0/ζx0)*inv(αβ0/ζβ0)*1/(p+(1-p)*1/v^2)
    return (;β0,βΔ0,a0)
  end

  function  βprior(::Val{:zerobetadelta0_2}; p,v,varβ, Eβ, kwargs...)
    β0=Eβ
    βΔ0=0.0

    #restrict the precision to a reasonable range
    Eτ=max(a0min, min(a0max, 1/varβ))
    Eτsansa0=αy0/ζy0*αx0/ζx0*αβ0/ζβ0*(p+(1-p)*1/v^2)
    a0=Eτ/Eτsansa0
    #a0raw=max(a0min, min(a0max, (p+(1-p)*v^2)/varβ))
    return (;β0,βΔ0,a0)
  end

  #=function βprior(::Val{:zerobetadelta0simulated}; p,v,varβ, Eβ, kwargs...) 
    
    β0=Eβ
    βΔ0=0.0

    τx = draw(:τx, simulateddraws; αx0=αx0, ζx=ζx0)
    τy = draw(:τy, simulateddraws; αy0=αy0, ζy=ζy0)
    τβ = draw(:τβ, simulateddraws; αβ0=αβ0, ζβ=ζβ0)
    γ = draw(:γ,p; simulateddraws; αβ0=αβ0, ζβ=ζβ0)

    #the below follows becuase
    #var(β)=1/(a0)*1/(τx*τy*τβ)*(γ .+ (1 .- γ)*v^2)), thus a0=E(1/(τx*τy*τβ)*(γ .+ (1 .- γ)*v^2)))/E(var(β)) (see model doc)
    meanifa0equals1 = mean(1/(τx.*τy.*τβ).*(γ .+ (1 .- γ)*v^2))
    a0raw = meanifa0equals1/varβ
    #also need to adjust the bounds: maxa0unadj=maxa0adj*E(1/(τx*τy*τβ)*(γ .+ (1 .- γ)*v^2)))/E(var(β))
    a0 = max(a0min/meanifa0equals1, min(a0max/meanifa0equals1, a0raw))
    return (;β0, βΔ0, a0)
  end=#

  function βprior(::Val{:knownbeta0};p,v,varβ, Eβ, β0, kwargs...)
    local βΔ0::Float64 = (Eβ-β0)/(p+v*(1-p))
    local a0::Float64 = (p+(1-p)*v^2)/(varβ - p*(β0+βΔ0)^2-(1-p)*(β0+v*βΔ0)^2 + Eβ^2)
    #this can happen as the above is based on a normal approx
    a0 = max(a0min, min(a0max, a0)).* inv(αy0/ζy0)*inv(αx0/ζx0)*inv(αβ0/ζβ0),
    #=if a0 ≤ 0.0
      @eval Main p=$p
      @eval Main β0=$β0
      @eval Main βΔ0=$βΔ0
      @eval Main a0=$a0
      @eval Main varβ=$varβ
      @eval Main Eβ=$Eβ
      throw("a0<0: a0=$a0")
    end=#
    (; β0,βΔ0,a0)
  end

  βinfo  = map(1:K) do k
    local β0::Float64 = eval(Meta.parse(dparam[(;param=Symbol(:β,"[$k]"))][begin,:prior])).β0 |> Float64
    local Eβ::Float64 = dparam[(;param=Symbol(:β,"[$k]"))][begin,:E,] |> Float64
    local varβ::Float64 = dparam[(;param=Symbol(:β,"[$k]"))][begin,:sd,]^2 |> Float64
    local p::Float64 = dparam[(;param=Symbol(:γ,"[$k]"))][begin,:E,]
    p = min(pγmax, max(pγmin, p)) |> Float64
    return (;β0, p, Eβ, varβ, )
  end
  
  βpriors = map(1:K) do k
    return βprior(Val{betapriormethod}(); v, βinfo[k]...)
  end
  #prior for β
  βΔ0 = βpriors .|> pr-> pr.βΔ0
  β0 = βpriors .|> pr->pr.β0
  A0 = (βpriors .|> pr->pr.a0) |> Diagonal

  if testhistpriors
    zerobetadelta0 = map(1:K) do k
      return βprior(Val{:zerobetadelta0}(); v, βinfo[k]...)
    end
    zerobetadelta0_2 = map(1:K) do k
      return βprior(Val{:zerobetadelta0_2}(); v, βinfo[k]...)
    end

    @eval Main zerobetadelta0=$zerobetadelta0
    @eval Main zerobetadelta0_2=$zerobetadelta0_2

    for k ∈ 1:K
      @assert zerobetadelta0[k].β0 ≈ zerobetadelta0_2[k].β0
      @assert zerobetadelta0[k].βΔ0 ≈ zerobetadelta0_2[k].βΔ0
      @assert zerobetadelta0[k].a0 ≈ zerobetadelta0_2[k].a0 "zerobetadelta0[k]).a0=$(zerobetadelta0[k].a0) ≈ zerobetadelta0_2[k].a0=$(zerobetadelta0_2[k].a0)"
    end
    @info "checked-zerobetadelta priros are equivalent"
  end

  #prior for ω
  if betapriormethod ≡ :knownbeta0
    #may want to make a heuristic here
    μω = dparam[(;param=:ω)][begin, :E]
  elseif betapriormethod ≡ :zerobetadelta0
    μω = dparam[(;param=:ω)][begin, :E]
  else
    throw("unknown βpriormethod $βpriormethod")
  end


  σ2ω = dparam[(;param=:ω)][begin, :sd]^2
  νω = μω*(1-μω)/σ2ω-1
  κ0 = μω*νω
  δ0 = (1-μω)*νω
  @assert  mean(Beta(κ0,δ0)) ≈ μω
  @assert var(Beta(κ0,δ0)) ≈ σ2ω

  ##DELETE THIS- for testing only

  
  hypert = (;
    ϕ = (;ϕ0, M0, ϕmin, ϕmax),
    β = (;v, β0, βΔ0, A0),
    τy = (;αy0, ζy0),
    τx = (;αx0, ζx0),
    τβ = (;αβ0, ζβ0),
    τϕ = (;αϕ0, ζϕ0),
    ν = (;αν0, ζν0, rνdist, lrν, νmin, νmax),
    ω = (; κ0, δ0,),
    x = (;),
    ψ = (;),
    γ = (;),
  )

  #override the histpriors with the default- helpful for testing
  for Fθ ∈ skiphistpriors
    θdef = getproperty(hyperdef,Fθ)
    #swap all priors associated with a particular parameter for the default
    if haskey(hypert, Fθ)
      hypert = merge(hypert, Dict(Fθ=>θdef))
      continue
    end

    #swap a single prior associated with a particular parameter for the default
    for k ∈ keys(hypert)
      if haskey(hypert[k],Fθ)
        hypert = merge(hypert, Dict(k=>merge(hypert[k],Dict(Fθ=>θdef))))
        continue
      end
    end

    throw("Did not recognize prior to skip $Fθ")
  end

  hyper = TΘ(; hypert...)

  return hyper
end

function loadpriors(priorset::Symbol, model; dims, data, kwargs...)
  priorparam = PARAM[priorset]
  if priorset ≡ :histpriors
    return loadpriorsfrompreviousresults(priorparam, model; dims, data, kwargs...)
  end
  
  return loadpriorsfromasciidict(priorparam, model; dims, data)

end

