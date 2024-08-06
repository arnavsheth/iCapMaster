draw(s::Symbol, args...; kwargs...)=draw(Val(s), args...;kwargs...)

draw(::Val{:y}, drawdims...; μy, 
  Λy=nothing, Σy=Λy |> pdinv) = rand(MultivariateNormal(μy, Σy),drawdims...) 

draw(::Val{:x}, drawdims...; μx, 
  Λx=nothing, Σx=Λx |> pdinv) = rand(MultivariateNormal(μx, Σx),drawdims...) 
draw(::Val{:β}, drawdims...; μβ,  Λβ=nothing, Σβ=Λβ |> pdinv) = rand(MultivariateNormal(μβ, Σβ),drawdims...) 
   

draw(::Val{:τx}, drawdims...; αx, ζx) = rand(Gamma(αx, 1/ζx),drawdims...) 
draw(::Val{:τy}, drawdims...; αy, ζy) = rand(Gamma(αy, 1/ζy),drawdims...) 
draw(::Val{:ω}, drawdims...; κ, δ) = rand(Beta(κ, δ),drawdims...) 

draw(::Val{:τβ}, drawdims...; αβ, ζβ) = rand(Gamma(αβ, 1/ζβ),drawdims...) 
draw(::Val{:τϕ}, drawdims...; αϕ, ζϕ) = rand(Gamma(αϕ, 1/ζϕ),drawdims...) 

function draw(::Val{:ϕ}, drawdims...; μϕ, 
  Λϕ=nothing, Σϕ=Λϕ |> pdinv, ϕmin, ϕmax,) 
  
  if (ϕmin ≡ nothing || ϕmin == -Inf) && (ϕmax≡nothing || ϕmax == Inf)
    rand(MultivariateNormal(μϕ, Σϕ),drawdims...) 
  else
    @assert isempty(drawdims) "repeated draws of truncated ϕ not supported (but could be)"
    
    truncatedmvn(μϕ, Σϕ; 
      lower=ϕmin, 
      upper=ϕmax, 
      Λ=Λϕ,)
  end 


end

draw(::Val{:ψt}; αψt, ζψt) = rand(Gamma(αψt,1/ζψt))
function draw(::Val{:ψ}, drawdims...; αψ, ζψ, T = length(αψ))
  @assert length(αψ) == length(ζψ)

  if isempty(drawdims) || (length(drawdims==1) && drawdims[1]==T)
    return ((αψt,ζψt, ::Any)->draw(Val{:ψt}(); αψt,ζψt)).(αψ,ζψ, 1:T)
  elseif drawdims[1] == T && length(drawdims)==2
    throw("Unsupported dimensions (but this is supportable- see the untested code here)")
    return reduce(vcat, rand(Gamma(αψt,1/ζψt), K, drawdims[2]...).(αψ,ζψ))
  else
    throw("Unsupported dimensions for draw ψ $drawdims")
  end
end

function draw(::Val{:γ}, drawdims...; pγ)
  K = length(pγ)

  if isempty(drawdims) || (length(drawdims==1) && drawdims[1]==K)
    return ((pγk)->rand(Bernoulli(pγk))).(pγ) |> Vector{Bool}
  else
    throw("Unsupported dimensions for draw ψ $drawdims")
  end
end

draw(::Val{:γ}, drawdim; pγ::Float64) = rand(Bernoulli(pγ),drawdim) |> Vector{Bool}


draw(::Val{:γ}, pγ::Float64) = rand(Bernoulli(pγ))

#this works a bit differently than the other draws as we need Metropolis-Hastings
function draw(::Val{:ν}, drawdims...; dgp, init=median(dgp.hyper.rνdist))
  @unpack rνdist, lrν = dgp.hyper 
  @unpack lpν = updateθpν(dgp)

  draws = imhsampler(drawdims...; lr=lrν, rdist=rνdist, lp=lpν, init)
  
  return draws
end

#single sample version
draw(::Val{:ν},; lpν, rνdist, lrν, νtM1) = imhsampler(; lr=lrν, rdist=rνdist, lp=lpν, ptM1=νtM1)

#an independence-style Metropolis Hastings sampler (independent proposal distributions)
function imhsampler(;lp,rdist,ptM1, lr=((x)->logpdf(rdist,x)))
  drawnval = rand(rdist)
  #prop=(p(drawnval)/r(drawnval))/(p(ptM1)/r(ptM1))
  prop=((lp(drawnval) - lr(drawnval)) - (lp(ptM1)- lr(ptM1))) |> exp
  (prop<1.0) && (rand(Uniform())>prop) && return ptM1
  
  return drawnval
end

function imhsampler(drawdims...;lp,rdist, init=0.0, lr= (x) -> logpdf(rdist,x) )
  draws = rand(rdist, drawdims...)
  
  ptM1 = init

  for (i,d) ∈ enumerate(draws)
    #prop=(p(d)/r(d))/(p(ptM1)/r(ptM1))
    prop = ((lp(d) - lr(d)) - (lp(ptM1)- lr(ptM1))) |> exp
    if (prop<1.0) && (rand(Uniform())>prop)
      draws[i] = ptM1
    else
      ptM1 = d
    end
  end

  return draws
end

function testimhsampler(;iter, verbose=true)

  #rdist = Uniform()

  rνdist=Gamma(αν0, 1/ζν0) #set proposal to prior
  lr(x)=logpdf(rdist,x) 
  
  pdist = Beta(2,6) #mean 0.25, var(0.0208)
  lp(x) = logpdf(pdist,x)
  init=0.5

  draws = imhsampler(iter; lp,rdist)

  drawsconsec = Vector{Float64}(undef, iter)
  ptM1 = median(rdist)
  for i ∈ 1:iter
    drawsconsec[i] = imhsampler(; lp,rdist, ptM1, lr)
    ptM1 = drawsconsec[i]
  end

  if verbose
    @info "Actual stats: mean=$(mean(pdist)), var=$(var(pdist)), skew=$(skewness(pdist)), kurt=$(kurtosis(pdist))"
    @info "Sample stats: mean=$(mean(draws)), var=$(var(draws)), skew=$(skewness(draws)), kurt=$(kurtosis(draws))"
    @info "Samples (consec) stats: mean=$(mean(drawsconsec)), var=$(var(drawsconsec)), skew=$(skewness(drawsconsec)), kurt=$(kurtosis(drawsconsec))"
  end
end



