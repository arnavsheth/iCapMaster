
#predict x conditional on F, β, and r only
predict(::Val{:ExF};
  dims, data, Θ,) =  data.F*Θ.β+data.r


#this version only predicts using the factors when there is no estimated value of x
#the main use case is for downstream best predictive draws or predictions of y (y and Ey)
function predict(::Val{:Ex}; dims, data, Θ, tstart = 1, testpredict=false)
  @unpack x = Θ
  @unpack T = dims
  
  #@eval Main x=$x
  Tx = length(x)


  
  #simplest case- predicting conditional on x
  if T==Tx
    @assert tstart==1
    return x
  end

  @unpack F,r = data
  tend = tstart+Tx-1


  xinds = tstart:(tstart+Tx-1)
  x̂ = Vector{Float64}(undef, T)
  x̂[xinds] .= x
  x̂[Not(xinds)] .= predict(Val{:ExF}(); dims=nothing, data=(;F=F[Not(xinds),:], r=r[Not(xinds)]), Θ)

  if testpredict
    x̂test = r + F*Θ.β
    x̂test[tstart:(tstart+length(x)-1)] .= x
    @assert x̂test ≈ x̂
  end

  return x̂
end

function predict(::Val{:Ey}; dims, data, Θ, tstart = 1, testpredict=false)
  @unpack ϕ = Θ
  x̂::Vector{Float64} = predict(Val{:Ex}(); dims, data, Θ, tstart, testpredict)

  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)

  ŷ = X̃L * ϕ .+ xS

  if testpredict
    @unpack F, r = data
    xinds = tstart:(tstart+length(Θ.x)-1)
    @assert all(x̂[xinds] .== Θ.x)

    Ftest = vcat(F[1:(tstart-1),:],F[(tstart+length(xinds)):end,:])
    rtest = vcat(r[1:(tstart-1)],r[(tstart+length(xinds)):end])

    @assert all(x̂[Not(xinds)] .≈ Ftest*Θ.β + rtest)
    @assert formΦ(ϕ; dims)*x̂ ≈ ŷ
  end

  return ŷ
end



#predict y conditional on F,β,r,ϕ only
function predict(::Val{:EyF},;dims,  data,  Θ,)
  @unpack ϕ = Θ


  x̂::Vector{Float64} = predict(Val{:ExF}(); Θ,data,dims)
  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)

  ŷ = X̃L * ϕ .+ xS

  return ŷ
end

#this provides a prediction using ExF for the most recent Δt period and x before that
function predict(::Val{:EyF1step}; dims, data, Θ, testpredict=false)
  @unpack x, ϕ = Θ
  Ex = predict(Val{:ExF}(); dims, data, Θ)

  
  ExS = formxS(Ex; dims)
  X̃L = formX̃L1step(; x, Ex, dims)
  ŷ = X̃L*ϕ+ExS

  if testpredict
    @unpack T,S,Δt,s2t, P, R = dims
    ϕ̃ = R*ϕ
    ŷtest = map(1:S) do s
      t = s2t[s]
      xt = [x[(t-Δt-P+1):(t-Δt)]; Ex[(t-Δt+1):t]]
      @assert length(xt) == P + Δt
      return xt'ϕ̃ + ExS[s]
    end

    @assert all((ŷtest .≈ ŷ) .| ((ŷtest.+1) .≈ (ŷ.+1)))
  end

  return ŷ
end


#this provides a prediction using xF for the most recent Δt period and x before that
function predictivedraw(::Val{:yF1step}; dims, data, Θ, testpredict=false)
  @unpack x, ϕ, τy = Θ
  @unpack S=dims
  x̂ = predictivedraw(Val{:xF}(); dims, data, Θ)

  
  xS = formxS(x̂; dims)
  X̃L = formX̃L1step(; x, Ex=x̂, dims)
  μy = X̃L*ϕ+xS

  if testpredict
    @unpack T,S,Δt,s2t, P, R = dims
    ϕ̃ = R*ϕ
    μytest = map(1:S) do s
      t = s2t[s]
      xt = [x[(t-Δt-P+1):(t-Δt)]; x̂[(t-Δt+1):t]]
      @assert length(xt) == P + Δt
      return xt'ϕ̃ + xS[s]
    end

    @assert all((μytest .≈ μy) .| ((μytest.+1) .≈ (μy.+1)))
  end



  Σy = fill(1/τy, S) |> Diagonal
  ŷ = draw(Val{:y}(); μy, Σy)

  return ŷ
end




function predict(::Val{:yx}; Θ, dims, data, testpredict=false)

  @unpack x,ϕ = Θ
  @unpack s2t, Δt = dims

  X̃L = formX̃L(; x, dims)
  xS = formxS(x; dims)
  ŷ = X̃L * ϕ .+ xS

  testpredict &&  @assert ŷ ≈ formΦ(ϕ;dims)*x

  return ŷ
end

#this version only predicts using the factors when there is no estimated value of x
#otherwise, residual variance is assumed 0 and x is treated as a conditioning quantity
#the main use case is for downstream best predictive draws of y
function predictivedraw(::Val{:x}; dims, data, Θ, tstart = 1, testpredict=false)
  @unpack x = Θ
  @unpack T = dims

  #degenerate case- no need to predict
  if length(x) == T
    @assert tstart==1
    return x
  end

  @unpack F,r = data

  Tx = length(x)

  xinds = tstart:(tstart+Tx-1)
  x̂ = Vector{Float64}(undef, T)
  x̂[xinds] .= x
  x̂[Not(xinds)] .= predictivedraw(Val{:xF}(); 
    dims=(;T=T-Tx), 
    data=(;F=F[Not(xinds),:], 
    r=r[Not(xinds)]), Θ)

  if testpredict
    @assert all(x̂[tstart:(tstart+Tx-1)] .== x)
  end

  return x̂

end

#create a stochastic predictive draw for x
function  predictivedraw(::Val{:xF};dims,  data,  Θ,)
  @unpack ν,τx,τy = Θ
  @unpack T = dims

  Ex = predict(Val{:ExF}(); Θ,data,dims)
  ψ̂ = draw(Val{:ψ}(); αψ= ν/2, ζψ=ν/2, T)

  Σx = inv.(ψ̂.*τx.*τy) |> Diagonal
  x̂ = draw(Val{:x}(); μx=Ex, Σx)

  return x̂
end

#create a stochastic predictive draw for y
function  predictivedraw(::Val{:yF};  dims,  data,  Θ,)
  @unpack S=dims
  @unpack τy,ϕ= Θ


  x̂ = predictivedraw(Val{:xF}(); Θ, data, dims)

  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)
  @assert length(xS) == size(X̃L,1) == S
  


  μy =  X̃L * ϕ .+ xS 

  Σy = fill(1/τy, S) |> Diagonal
  ŷ = draw(Val{:y}(); μy, Σy)

  return ŷ
end

#this version predicts from x when available, otherwise from xF
function predictivedraw(::Val{:y}; dims, data, Θ, tstart = 1, testpredict=false)

  @unpack τy, ϕ = Θ
  @unpack S = dims

  x̂::Vector{Float64} = predictivedraw(Val{:x}(); Θ, data, dims, tstart, testpredict)

  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)
  @assert length(xS) == size(X̃L,1) == S  

  μy =  X̃L * ϕ .+ xS 

  Σy = fill(1/τy, S) |> Diagonal
  ŷ = draw(Val{:y}(); μy, Σy)

  if testpredict
    @assert all(x̂[tstart:(tstart+length(Θ.x)-1)] .== Θ.x)
    @assert formΦ(ϕ; dims)*x̂ ≈ μy
  end

  return ŷ
end

#####--tests below


derived(V::Val{:xF_full_test},args...; includecumsumstat=false, kwargs...)=derived(
  Val{:xF_test}(), args...;  seriesstattype=V, includecumsumstat, kwargs...)

derived(V::Val{:xF_full_cum_test},args...; kwargs...)=derived(
    Val{:xF_full_test}(), args...; includecumsumstat=true, kwargs...)


derived(V::Val{:yF_full_test},args...; includecumsumstat=false, kwargs...)=derived(
  Val{:yF_test}(), args...; includecumsumstat, seriesstattype=V, kwargs...)

derived(V::Val{:yF_full_cum_test},args...; kwargs...)=derived(
  Val{:yF_full_test}(), args...; includecumsumstat=true, kwargs...)

function derived(V::Val{:EyF_test}, dgp::AbstractDGP; testderived=false)
  @unpack dims, Θ, data =dgp
  @unpack ϕ,β = Θ
  @unpack s2t, Δt = dims
  @unpack F,r,y = data

  x̂ = F*β + r
  X̃L = formX̃L(; x=x̂, dims)
  xS = formxS(x̂; dims)

  ŷ = X̃L * ϕ .+ xS

  testderived &&  @assert (ŷ ≈ formΦ(ϕ;dims)*x̂)

  return seriesstats(V, dgp;est=ŷ, act=y)
end
function derived(V::Val{:yF_test}, dgp::AbstractDGP; 
  data=dgp.data,
  dims=dgp.dims,
  includecumsumstat=false,
  seriesstattype=V,
  kwargs...)
  @unpack Θ =dgp
  @unpack ϕ,β,ν, τy, τx = Θ
  @unpack T,S = dims
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
  ŷ = μy + rand(Normal(0.0,1/τy^0.5),S)

 
  stats = (typeof(data) <: Data ? 
    seriesstats(seriesstattype, dgp;est=ŷ, act=data.y,includecumsumstat) :  
    seriesstats(seriesstattype, dgp;est=ŷ, includecumsumstat))


  return stats
end



function derived(V::Val{:ExF_test}, dgp::AbstractDGP; 
  data=dgp.data,
  dims=dgp.dims,
  testderived=false, 
  seriesstattype=V,
  includecumsumstat = false)


  @unpack T = dims
  @unpack Θ=dgp
  @unpack β = Θ

  x̂ = data.F*β + data.r

    
  return seriesstats(seriesstattype, dgp;est=x̂, includecumsumstat)
end

function derived(V::Val{:xF_test}, dgp::AbstractDGP; 
  data=dgp.data,
  dims=dgp.dims,
  testderived=false, 
  seriesstattype=V,
  includecumsumstat = false)


  @unpack T = dims
  @unpack Θ=dgp
  @unpack ν,β,τx,τy = Θ
  @unpack F,r = data

  μx = F*β+r
  ψ = draw(Val{:ψ}(); αψ= ν/2, ζψ=ν/2, T )


  x̂ = map(μx, ψ) do μxt, ψt
    rand(Normal(μxt, sqrt(inv(ψt*τx*τy))))
  end
    
  return seriesstats(seriesstattype, dgp;est=x̂, includecumsumstat)
end
