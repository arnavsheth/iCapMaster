#NOTE- the below will need to be refactored when we get serious

using Distributions, Statistics, LinearAlgebra, Random, DataFrames, StatsBase, BenchmarkTools, UnPack, SparseArrays
import Base: Symbol, show, length, @kwdef
import emf_gmam_ae as m


function testnested(;T,K, P, iter)

  Σx = rand(Normal(), T,K)*rand(K,K) |> cov
  ΣL = rand(Normal(), T, K) *rand(K,K) |> cov
  μx = zeros(K)
  μL = zeros(K) 

  EM = T*(ΣL + μL*μL')
  ExtMx = tr(EM*Σx)+μx'*EM*μx
  resLtL =  0.0#zeros(iter)
  resxtLtLx= 0.0#zeros(iter)
  for i ∈ 1:iter
    L = rand(MultivariateNormal(μL, ΣL), T)'
    x = rand(MultivariateNormal(μx,Σx))
    #resLtL[i] = sum(L'*L)
    #resxtLtLx[i] = x'*L'*L*x
    resLtL += sum(L'*L)
    resxtLtLx += x'*L'*L*x
  end

  @info "simmed LtL: $(mean(resLtL)/iter); act: $(EM |> sum)"
  @info "simmed xtMx: $(mean(resxtLtLx)/iter); act: $(ExtMx)"
end

#=
Incompelte test for restricted dimensions
function testrestricted(;dims)
  @unpack T, S,P,s2Lts = dims
  Σϕ̃ = rand(Normal(),T,K-1)*rand(K-1,K-1) |> cov
  ϕ̃ = rand(3)
  Σϕ = [[ϕ̃'*Σϕ̃*ϕ̃ ϕ̃'*Σϕ̃]; [Σϕ̃ * ϕ̃ Σϕ̃]]

  ϕ = [1-sum(ϕ̃); ϕ̃]
  Φ = formΦ(ϕ; dims)

  EΦtΦ .= zeros(T,T)
  M̃ = Σϕ .+ μϕ*μϕ'

  for s ∈ 1:S
    EΦtΦ[s2Lts[s],s2Lts[s]] .+= M̃
  end

  return EΦtΦ

end
=#

#generate the S×P matrix XL (using the sparse matrices)
function formXLfromx_alt!(x::AbstractVector{Tx}, XL::AbstractMatrix{Tx}; dims) where Tx
  
  
  #sanity check on dimensionality

  @unpack T,S,P, s2Lts, s2maxt, ιXL = dims
  @assert T == length(x) "T: $T, length(x): $(length(x)) x: $x"
  @assert (S,P) ≡ size(XL)
  @assert length(ιXL) ≡ P

  #XLexpanded = dropdims(sum(ιXL .* x', dims=2),dims=2)
  #@info "got here"
  #XLexpanded = reduce(hcat, (ιXL .|> ιXLp->Threads.@spawn(ιXLp*x)) .|> fetch)
  XLexpanded = reduce(hcat, (ιXL .|> ιXLp->ιXLp*x))

  XL=XLexpanded[s2maxt,:]

  #@info "got here"

  return XL
end

formXLfromx_alt(x::AbstractVector; dims,) = formXLfromx_alt!(x, zeros(dims.S, dims.s2Lts[1] |> maximum); dims)


#Reference the model doc- this is the second moment matrix for ϕ padded by zeros
#Easiest to construct this by summing the expectations of the outer product of the rows of Φ
function formEΦtΦ!(μϕ::AbstractVector, Λϕ::AbstractMatrix, EΦtΦ::AbstractMatrix;dims)
  @unpack S, T, P, s2Lts = dims
  
  EΦtΦ .= 0.0
  M̃ = inv(Λϕ) .+ μϕ*μϕ'

  for s ∈ 1:S
    EΦtΦ[s2Lts[s],s2Lts[s]] .+= M̃
  end

  return EΦtΦ
end

formEΦtΦ(μϕ::AbstractVector, M::AbstractMatrix, ; dims) = formEΦtΦ!(μϕ, M, zeros(dims.T, dims.T); dims)

formEXL(μx::AbstractVector; dims) = m.formXL(μx; dims)

#Reference the model doc- this is the second moment matrix for ϕ padded by zeros
#Easiest to construct this by summing the expectations of the outer product of the rows of Φ
function testformEΦtΦ!(μϕ::AbstractVector, Λϕ::AbstractMatrix, EΦtΦ::AbstractMatrix;dims)
  @unpack S, T, P, s2maxt = dims
  
  EΦtΦ .= 0.0
  M̃ = inv(Λϕ) .+ μϕ*μϕ'

  for s ∈ 1:S
    
    EΦsΦst = zeros(T,T)
    UL = zeros(s2maxt[s]-P,s2maxt[s]-P)
    EΦsΦst[1:(s2maxt[s]-P),1:(s2maxt[s]-P)] .= UL
    EΦsΦst[(s2maxt[s]-P+1):s2maxt[s],(s2maxt[s]-P+1):s2maxt[s]] .= M̃
    if t[s] < T
      LR = zeros(T-s2maxt[s],T-s2maxt[s])
      EΦsΦst[(s2maxt[s]+1):end,(s2maxt[s]+1):end] .= LR
    end
  end

  @assert formEΦtΦ(μϕ,Λϕ; dims)

  return EΦtΦ
end



function formEXLtXL!(μx::AbstractVector,Λx::AbstractMatrix, EXLtXL::AbstractMatrix; dims)
  @unpack S,T,P, ιXL =dims


  local Σx
  try 
    Σx = cholesky(Symmetric(Λx)) |> inv
  catch err
    @warn "Σx factorization failed ($err)! switching to pinverse"
    Σx = Symmetric(Λx) |> pinv
  end
  handles = Matrix(undef, P, P)

  # NOTE: the below version is equivelent and simpler but slower
  #=Threads.@threads for r ∈ 1:P
    Threads.@threads for c ∈ 1:P
      (r < c) && continue

      EXLtXL[r,c] = tr(ιXL[r]*Σx*ιXL[c]') + μx'*ιXL[r]' *ιXL[c]*μx

      #take advantage of symmetry
      if r > c
        EXLtXL[c,r] = EXLtXL[r,c]
      else
        @assert r==c      
      end
    end
  end=#

  #this is the fastest approach
  ΣxιXLt = [Σx * ιXL[p]'  for p ∈ 1:P]
  for r ∈ 1:P
    for c ∈ 1:P
      (r < c) && continue
    
      handles[r,c] = Threads.@spawn tr(ιXL[r] *ΣxιXLt[c]) + μx' * ιXL[r]' * ιXL[c] * μx

      #take advantage of symmetry
      if r > c
        handles[c,r] = handles[r,c]
      else
        @assert r==c
      end
    end
  end

  EXLtXL .= handles .|> fetch


  return EXLtXL

end

formEXLtXL(μx::AbstractVector,Λx::AbstractMatrix, ; dims, kwargs...)= formEXLtXL!(μx,Λx, Matrix{Float64}(undef, dims.P, dims.P); dims, kwargs...)


function testformEXLtXL(μx::AbstractVector,Λx::AbstractMatrix; dims)
  @unpack S,T,P,Δt, ιXL, s2maxt =dims  

  ιXLcheck = [falses(T,T) for p ∈ 1:P]
  for s ∈ 1:S, p ∈ 1:P
    ιXLcheck[p][s2maxt[s], s2maxt[s]-P+p] = true
  end

  for (ιXLcheckp, ιXLp )∈ zip(ιXLcheck, ιXL)
    @assert all(ιXLcheckp .== ιXLp)
  end

  EXLtXLcheck = Matrix{Float64}(undef, dims.P, dims.P)
  Σx = svd(Λx)\I

  for r ∈ 1:P, c∈1:P
    EXLtXLcheck[r,c] = tr(ιXLcheck[r]*Σx*ιXLcheck[c]')+μx'*ιXLcheck[r]'*ιXLcheck[c]*μx
  end

  @assert all(formEXLtXL(μx,Λx; dims) .≈ formEXLtXL!(μx,Λx,rand(P,P); dims) )
  @assert all(formEXLtXL!(μx,Λx,rand(P,P); dims) .≈ EXLtXLcheck)

  @info "check on testformEXLtXL successful for (S,T,P,Δt)= $((S,T,P,Δt)) complete"


end



function testΦXL(;S=20, samplesize=10^3, testmomentsEΦtΦ=true, testmomentsEXLtXL=true)
  m.tests2Lts()

  #test a simple single Δt case first
  Δt=1
  P=6
  K=4
  T = P + (S-1)*Δt
  x = rand(T)
  ϕ = rand(P)
  dims=m.Dims(;S,P, K, Δt, testdims=true)
  @unpack s2maxt, s2Lts = dims

  #sanity checks on the Δt mapping
  @assert s2maxt[1] == P
  @assert s2maxt[S] == T == maximum(s2Lts[end])
  XL = m.formXL(x; dims)
  Φ = m.formΦ(ϕ; dims)
  @assert all( Φ*x .≈ XL * ϕ)
  
  ###now test the quarterly case
  #test a simple single Δt case first
  Δt=3
  P=6
  T = P + (S-1)*Δt
  x = rand(T)
  ϕ = rand(P)
  dims=m.Dims(;S,P, K, Δt, testdims=true)
  @unpack s2maxt, s2Lts = dims

  @assert s2maxt[1] == P
  @assert s2maxt[S] == T == maximum(s2Lts[end])
  XL = m.formXL(x; dims)
  Φ = m.formΦ(ϕ; dims)
  @assert all( Φ*x .≈ XL * ϕ) 

  #now test the moments
  if testmomentsEΦtΦ
    B = rand(Normal(),P,P) #includes a factor structure
    P1s = ones(P)
    ϕs = rand(Normal(1.0,1.0), samplesize, P) * B
    μϕ = (P1s'*B) |> vec
    Σf = I(P) #this works because the factors are independent with variance 1
    Σϕ = B'*Σf * B
    ΦtΦtot = zeros(T,T)
    for i ∈ 1:samplesize
      Φ = formΦfromϕ(ϕs[i,:]; dims)
      ΦtΦtot .+= Φ'*Φ
    end

    
    simEΦtΦ = ΦtΦtot ./ samplesize
    EΦtΦ = formEΦtΦ(μϕ, Σϕ |> inv; dims)
    if S≤20
      @info "for P=$P and Δt=$Δt"
      @info "simulated EΦtΦ"
      display(simEΦtΦ)
      @info "calculated EΦtΦ"
      display(EΦtΦ)
    end

    #hard to check if most of the values are close enough but we can at least check the 0s
    @assert all((EΦtΦ .== simEΦtΦ .== 0.0) .| ((EΦtΦ .≠ 0.0) .&  (simEΦtΦ .≠ 0.0)))
  end

  #now test the moments of XL


  if testmomentsEXLtXL
    #####First the simple case
    Δt=1
    P=6
    T = P + (S-1)*Δt
    dims= m.Dims(;P,S,K,Δt, testdims=true)
    @assert dims.T == T

    #create a heteroskedastic and correlated sample set for X
    σ2x = rand(Normal(), T) .^2 .+ 0.1
    xsseed = rand(MultivariateNormal(zeros(T),Diagonal(σ2x)), samplesize)'
    @assert size(xsseed) ≡ (samplesize,T)
    Wxseed = rand(Normal(),T,T)
    μxseed = rand(Normal(),T)
    xs = eachrow((xsseed .+ μxseed') * Wxseed) .|> vec
    μx = Wxseed'*μxseed 

    #the below works because each col of W is a set of weights of xsseed, 
    #for which we have the covariance matrix
    Σx = Wxseed'*Diagonal(σ2x) * Wxseed
    Λx = Σx\I


    simEXL = mean(xs .|> xi->m.formXL(xi; dims))
    EXL = formEXL(μx; dims)


    simEXLtXL = mean(((xs .|> xi->m.formXL(xi; dims)) .|> XLi->XLi'*XLi))
    EXLtXL = formEXLtXL(μx, Λx; dims)

    @btime formEXLtXL!($μx, $Λx, $EXLtXL; dims=$dims)
    
    basicactiveinds = [zeros(Bool, P-1); ones(Bool, T-P+1)]
    μxs = mean(xs)

    if S≤20
      @info "for S=$S, T=$T, and Δt=$Δt"


      @assert length(basicactiveinds) == T
      @info "simulated EXLtXL"
      display(simEXLtXL)
      @info "calculated EXLtXL"
      display(EXLtXL)
    end
    testformEXLtXL(μx,Λx; dims)


    #####More compelx case
    Δt=3
    P=12
    K=4
    T = P + (S-1)*Δt
    dims= m.Dims(;K,P,S,Δt, testdims=true)
    @assert dims.T == T

    #create a heteroskedastic and correlated sample set for X
    σ2x = rand(Normal(), T) .^2 .+ 0.1
    xsseed = rand(MultivariateNormal(zeros(T),Diagonal(σ2x)), samplesize)'
    @assert size(xsseed) ≡ (samplesize,T)
    Wxseed = rand(Normal(),T,T)
    μxseed = rand(Normal(),T)
    xs = eachrow((xsseed .+ μxseed') * Wxseed) .|> vec
    μx = Wxseed'*μxseed 

    #the below works because each col of W is a set of weights of xsseed, 
    #for which we have the covariance matrix
    Σx = Wxseed'*Diagonal(σ2x) * Wxseed
    Λx = Σx\I


    simEXL = mean(xs .|> xi->m.formXL(xi; dims))
    EXL = formEXL(μx; dims)


    simEXLtXL = mean(((xs .|> xi->m.formXL(xi; dims)) .|> XLi->XLi'*XLi))
    EXLtXL = formEXLtXL(μx, Λx; dims)

    @btime formEXLtXL!($μx, $Λx, $EXLtXL; dims=$dims)
    
    

    basicactiveinds = [zeros(Bool, P-1); ones(Bool, T-P+1)]
    μxs = mean(xs)

    if S≤20
      @info "for S=$S, T=$T, and Δt=$Δt"


      @assert length(basicactiveinds) == T
      @info "simulated EXLtXL"
      display(simEXLtXL)
      @info "calculated EXLtXL"
      display(EXLtXL)
    end
    testformEXLtXL(μx,Λx; dims)

  end

end



@time testΦXL(;S=10, samplesize=10^3, testmomentsEΦtΦ=false, testmomentsEXLtXL=true)


