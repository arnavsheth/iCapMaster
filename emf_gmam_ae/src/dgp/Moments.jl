
#helper functions for computing the moments
#In some cases, an optimized version is provided which takes in a helper
#allocation or intermediate result

moment(s::Symbol, args...; kwargs...)=moment(Val{s}(), args...; kwargs...)



moment(V::Val{:EXL}; dims, μx) = moment!(V, Matrix{Float64}(undef, dims.S, dims.P+dims.Δt); dims, μx,)
moment!(::Val{:EXL}, EXL; dims, μx) = _formXL!(μx, EXL; dims)

moment(V::Val{:EX̃L}; dims,EXL) = moment!(V, Matrix{Float64}(undef, dims.S, dims.P); dims, EXL)
moment!(::Val{:EX̃L}, EX̃L; dims, EXL) = formX̃L!(EX̃L; XL=EXL, dims)

moment(::Val{:ExS}; μx, dims) = formxS(μx; dims)

moment(::Val{:Eỹ}; μx, y, dims) = formỹ(;y, x=μx, dims)

function moment(::Val{:Eỹtỹ}; EXLtXL, dims, y, μx=nothing, ExS=moment(Val{:ExS}(); μx, dims))
  @unpack s2t, ιXL, P, Δt = dims
  
  return sum(EXLtXL[(P+1):end,(P+1):end])+y'y -2*y'*ExS
end
  

#computes the expectation E[XL'XL]
function moment!(::Val{:EXLtXL}; μx::AbstractVector,
    Σx::AbstractMatrix, EXLtXL::AbstractMatrix, dims)
  @unpack S,T,P, ιXL, Δt =dims

  #this is the fastest approach
  handles = Matrix(undef, P+Δt, P+Δt)
  ΣxιXLt = [Σx * ιXL[p]'  for p ∈ 1:(P+Δt)]
  for r ∈ 1:(P+Δt)
    for c ∈ 1:(P+Δt)
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

moment(V::Val{:EXLtXL}; μx, Σx, dims) = moment!(V; μx, Σx, dims, EXLtXL=Matrix{Float64}(undef, dims.P+dims.Δt, dims.P+dims.Δt))


moment(::Val{:EX̃LtX̃L}; dims,  EXLtXL) = (dims.R' * EXLtXL * dims.R)

moment(::Val{:EX̃LtxS}; EXLtXL, dims) = dims.R'vec(sum(EXLtXL[:,(dims.P+1):end], dims=2))


moment(::Val{:EX̃Ltỹ}; EXL, EXLtXL, y, dims) = dims.R'EXL'*y - moment(Val{:EX̃LtxS}(); EXLtXL,dims)



#two paths- 1) provide XLtXL, then EX̃LtX̃L will ve caculated
# 2) provide EX̃LtX̃L
moment(::Val{:EϕtX̃LtX̃Lϕ}; μϕ, 
  Σϕ,
  EXLtXL=nothing,
  dims=nothing,
  EX̃LtX̃L::AbstractMatrix=moment(Val{:EX̃LtX̃L}(); EXLtXL, dims),
) = tr(EX̃LtX̃L*Σϕ)+μϕ'*EX̃LtX̃L*μϕ

#=function moment(V::Val{:EϕtX̃LtX̃Lϕ}; EXLtXL=nothing, , dims)

  EX̃LtX̃L=moment(Val{:EX̃LtX̃L}(); EXLtXL, dims)
  return moment(V, EX̃LtX̃L; μϕ, Σϕ)
end=#



moment(::Val{:Eτy}; αy, ζy) = αy/ζy
moment(::Val{:Eτx}; αx, ζx) = αx/ζx
moment(::Val{:Eτϕ}; αϕ, ζϕ) = αϕ/ζϕ
moment(::Val{:Eτβ}; αβ, ζβ) = αβ/ζβ

moment(::Val{:Eψ}; αψ, ζψ) = αψ ./ ζψ
moment(::Val{:Elogψ}; αψ, ζψ) = digamma.(αψ) .- log.(ζψ)
moment(::Val{:EΨ}; αψ=nothing, ζψ=nothing, Eψ=moment(Val{:Eψ}(); αψ, ζψ)) = Eψ  |> formΨ

moment(::Val{:ExtΨx}; μx, Σx, EΨ) = tr(EΨ*Σx)+μx'*EΨ*μx
moment(::Val{:Ex2}; μx, Σx) = μx .^2 + diag(Σx)

_moment(::Val{:EβtFtΨFβ}; μβ, Σβ, F, EΨ) = tr(F'*EΨ*F*Σβ)+μβ'*F'*EΨ*F*μβ
moment(::Val{:EβtFtPrtΨFβPr}; μβ, Σβ, F, EΨ, r)=tr(F'*EΨ*F*Σβ)+(μβ'*F'+r')*EΨ*(F*μβ+r)
moment(::Val{:EβtFtPrT12tΨFβPrT12}; μβ, Σβ, F, EΨ, r,dims)=tr(F'*EΨ*F*Σβ)+(μβ'*F'+sqrt(dims.T)*r')*EΨ*(F*μβ+sqrt(dims.T)*r)



_moment(::Val{:Eβtfftβ}; μβ, Σβ, F) = eachrow(F) .|> f->tr(f*f'*Σβ)+μβ'*f*f'*μβ
moment(::Val{:EβtfPrftβPr}; μβ, Σβ, F,r) = ((f,rt)->tr(f*f'*Σβ)+(μβ'*f+rt)*(f'*μβ+rt)).(eachrow(F), r)

moment(::Val{:EβtfPrT12ftβPrT12}; μβ, Σβ, F,r,dims) = ((f,rt)->tr(f*f'*Σβ)+(μβ'*f+rt)*(f'*μβ+rt)).(eachrow(F), r*sqrt(dims.T))


moment(::Val{:EDA0D}, A0::Diagonal; pγ, v, a0=A0.diag) = Diagonal(a0 .* pγ .+ a0 .* (1.0 .- pγ)./v^2)
function moment(::Val{:EDA0D}, A0::Matrix; pγ, v, ED, 
  K=length(pγ),
  EDA0D=Matrix{Float64}(undef, K, K))

  Ed = ED.diag 

  #the below is valid for everything but the diagonals
  EDA0D .= Ed * Ed'

  #fix the diagonals
  for r ∈ 1:K
    EDA0D[r,r] = pγ[r] + (1.0 - pγ[r]) / v^2
  end
  

  EDA0D .= Symmetric(EDA0D .* A0)


  return EDA0D
end

moment(V::Val{:EDA0D}; A0, kwargs...) = moment(V,A0; kwargs...)


#moment(::Val{:EDA0D}; A0, pγ, v, ED, dims,) = moment(Val{:EDA0Dgdk}(); A0, pγ, v, ED, dims, dk=nothing, k=0,)

moment(::Val{:ED}; pγ, v,) = Diagonal(pγ .+ (1.0 .- pγ)./v)

moment(::Val{:EβtDA0Dβ};  μβ, Σβ, EDA0D) = tr(EDA0D*Σβ)+μβ'*EDA0D*μβ

moment(::Val{:Eβ̃}; μβ, β0) = μβ-β0
moment(::Val{:Eβ̃tDA0Dβ̃}; EDA0D, Eβ̃, Σβ) = moment(Val{:EβtDA0Dβ}(); μβ=Eβ̃,Σβ, EDA0D)



#moment(::Val{:Eβ2}; Σβ, μβ, σ2β=Σβ |> diag)= σ2β .+ μβ.^2

moment(::Val{:Eω}; κ,δ) = κ/(κ + δ)
moment(::Val{:Elogω}; κ,δ) = digamma(κ) - digamma(κ+δ)
moment(::Val{:Elog1Mω}; κ,δ) = digamma(δ) - digamma(κ+δ)

#Reference the model doc- this is the second moment matrix for ϕ padded by zeros
#Easiest to construct this by summing the expectations of the outer product of the rows of Φ
function moment!(::Val{:EΦtΦ}; μϕ, Σϕ, EΦtΦ, dims)
  @unpack S, T, P, Δt, s2allts, ιϕ = dims
  
  EΦtΦ .= 0.0
  Eϕϕt = Σϕ + μϕ*μϕ'

  Δt1s = ones(Δt)
  Eϕ1Mϕtιϕ = μϕ*Δt1s'-Eϕϕt*ιϕ
  EιtϕΔt1st = ιϕ'μϕ*Δt1s'
  E1Mϕtιϕ1Mϕtιϕt = Δt1s*Δt1s' - EιtϕΔt1st - EιtϕΔt1st' + ιϕ'Eϕϕt*ιϕ
  M̃ = [Eϕϕt Eϕ1Mϕtιϕ; transpose(Eϕ1Mϕtιϕ)  E1Mϕtιϕ1Mϕtιϕt]

  for s ∈ 1:S
    EΦtΦ[s2allts[s],s2allts[s]] .+= M̃
  end

  return EΦtΦ
end

moment(::Val{:EΦtΦ}; μϕ::AbstractVector, Σϕ::AbstractMatrix,  dims) = moment!(Val{:EΦtΦ}(); μϕ, Σϕ, EΦtΦ=zeros(dims.T, dims.T), dims)

moment!(::Val{:EΦ}; EΦ, dims, μϕ) = formΦ!(μϕ, EΦ; dims)
moment(V::Val{:EΦ}; dims, μϕ)= moment!(V; dims, μϕ, EΦ=Matrix{Float64}(undef, dims.S, dims.T))

moment(::Val{:EϕtM0ϕ}; μϕ, Σϕ, M0)=tr(M0*Σϕ) + μϕ'*M0*μϕ



#not technically a moment function- needed to get the normalization coefficient for the pdf

moment(::Val{:η2};pν::Function, νmin, νmax , rtol=PARAM[:integralrtol]) = 1/quadgk(pν, νmin, νmax; rtol)[1]

#Default of e0eps() sets the integration precision at one decimal point less than machine precision
moment(::Val{:Eν};pν::Function, νmax, νmin, 
  η2=moment(:η2; pν, νmax, νmin),
  rtol=PARAM[:integralrtol],
  ) = quadgk((ν)->ν*pν(ν), νmin, νmax; rtol)[1]*η2
  
#note- the below fixed method seems to be slower than the adaptive approach above
#wrapper function for creating integration nodes
#I found the pre-computed node approach to be slower
moment(::Val{:Eν}, nodes;pν::Function, ) = dot(nodes.νw, pν.(nodes.ν))/dot(nodes.w, pν.(nodes.ν))
function computeνnodes(f, fargs...; νmin, νmax, rawνmin=-1.0, rawνmax=1.0, rescale=isfinite(νmax-νmin))
  νraw, wraw = f(fargs...)
  ν = rescale ? (νraw .- rawνmin) .* (νmax-νmin)/(rawνmax-rawνmin) .+ νmin : νraw
  w = rescale ? wraw .* (νmax-νmin)/(rawνmax-rawνmin) : wraw
  νw = ν .* w

  return (;ν, w, νw)
end


#####################simple tests for the more complex moments#################



function testmoment(V::Val{:EβtfPrftβPr}; μβ, Λβ, F,r)
  Σβ = Λβ |> pdinv


  EβtfPrftβPr = moment(V; μβ, Σβ, F,r)
  for (f,EβtfPrftβtPrt,rt) ∈ zip(eachrow(F),EβtfPrftβPr,r)
    @assert tr(f*f'*pdinv(Λβ)) + μβ'*f*f'*μβ + 2*μβ'*f*rt+rt^2 ≈ EβtfPrftβtPrt
  end

  return nothing
end


function testmoment(V::Val{:EβtfPrT12ftβPrT12}; μβ, Λβ, F, r, dims)
  Σβ = Λβ |> inv
  @unpack T=dims

  #@assert moment(V; μβ, Λβ, F, EΨ) ≈ moment(V; μβ, Σβ, F, EΨ)
  EβtfPrT12ftβPrT12 = moment(V; μβ, Σβ, F,r,dims)
  for (f,EβtfPrT12ftβPrT12t,rt) ∈ zip(eachrow(F),EβtfPrT12ftβPrT12,r)
    @assert tr(f*f'*pdinv(Λβ)) + μβ'*f*f'*μβ + 2*μβ'*f*rt*sqrt(T)+rt^2*T ≈ EβtfPrT12ftβPrT12t
  end
  @assert EβtfPrT12ftβPrT12 ≈ moment(Val{:EβtfPrftβPr}(); μβ, Σβ, F, r=r*sqrt(T),)

  return nothing
end

function testmoment(::Val{:Eν}; lpν, νmin, νmax, αν0, ζν0)
  pν(x) = exp(lpν(x))

  @assert 1.0 ≈ moment(:η2; pν, νmin, νmax)*quadgk(pν,νmin, νmax)[1]

  νmax = min(νmax, quantile(Gamma(αν0, 1/ζν0), 1-eps()))
  nodes = computeνnodes(gausslobatto, 10^5; νmin, νmax)
  Eν = moment(:Eν;pν, νmin, νmax)
  Eνcheck = moment(:Eν, nodes; pν)
  @assert Eν ≈ Eνcheck
end

function testmoment(::Val{:EΦtΦ}; μϕ::AbstractVector, Λϕ::AbstractMatrix,dims)
  @unpack S, T, P, s2t, ιϕ, Δt = dims
 

  #Build EΦtΦ from scratch
  M̃ = zeros(P+Δt,P+Δt)

  Σϕ = Λϕ |> inv
  EΦtΦ = zeros(T,T)
  Eϕϕt = Σϕ + μϕ*μϕ'

  for i ∈ 1:(P+Δt), j∈1:(P+Δt)
    if i ≤ P && j ≤ P
      M̃[i,j] = Σϕ[i,j] + μϕ[i]*μϕ[j]
    elseif i ≤ P && j > P
      M̃[i,j] = μϕ[i] - Eϕϕt[i,:]'*ιϕ[:,j-P]
    elseif i > P && j ≤ P
      M̃[i,j] = μϕ[j] - Eϕϕt[j,:]'*ιϕ[:,i-P]
    elseif i > P && j > P
      M̃[i,j] = 1 - μϕ'ιϕ[:,j-P]-ιϕ[:,i-P]'μϕ+ιϕ[:,i-P]'Eϕϕt*ιϕ[:,j-P]
    else
      @assert false
    end
  end

  #Now accumulate the M̃ matrices
  for s ∈ 1:S
    
    EΦsΦst = zeros(T,T)
    UL = zeros(s2t[s]-P-Δt,s2t[s]-P-Δt)
    EΦsΦst[1:(s2t[s]-P-Δt),1:(s2t[s]-P-Δt)] .= UL
    EΦsΦst[(s2t[s]-P-Δt+1):s2t[s],(s2t[s]-P-Δt+1):s2t[s]] .= M̃
    if s2t[s] < T
      LR = zeros(T-s2t[s],T-s2t[s])
      EΦsΦst[(s2t[s]+1):end,(s2t[s]+1):end] .= LR
    end

    EΦtΦ .+= EΦsΦst
  end

  #@eval Main test=$EΦtΦ
  #@eval Main mom=$(moment(Val{:EΦtΦ}(); dims, μϕ, Λϕ))

  @assert EΦtΦ ≈  moment(Val{:EΦtΦ}(); dims, μϕ, Σϕ) 
  @assert EΦtΦ ≈ moment!(Val{:EΦtΦ}(); dims, μϕ, Σϕ, EΦtΦ=deepcopy(EΦtΦ))

  return
end


function testmoment(::Val{:EX̃LtX̃L}; μx::AbstractVector,Λx::AbstractMatrix, dims)
  @unpack S,T,P,Δt, ιXL, s2t, R =dims  

  ιXLcheck = [falses(S,T) for p ∈ 1:(P+Δt)]
  for s ∈ 1:S, p ∈ 1:(P+Δt)
    ιXLcheck[p][s, s2t[s]-P+p-Δt] = true
  end

  ιXLcheckexpanded = [falses(T,T) for p ∈ 1:(P+Δt)]
  for s ∈ 1:S, p ∈ 1:(P+Δt)
    ιXLcheckexpanded[p][s2t[s], s2t[s]-P+p-Δt] = true
  end

  for (ιXLcheckp, ιXLp )∈ zip(ιXLcheck, ιXL)
    @assert all(ιXLcheckp .== ιXLp)
  end

  EXLtXLcheck = Matrix{Float64}(undef, dims.P+Δt, dims.P+Δt)
  Σx = svd(Λx)\I

  for r ∈ 1:(P+Δt), c∈1:(P+Δt)
    EXLtXLcheck[r,c] = tr(ιXLcheck[r]*Σx*ιXLcheck[c]')+μx'*ιXLcheck[r]'*ιXLcheck[c]*μx
    @assert EXLtXLcheck[r,c] ≈ tr(ιXLcheckexpanded[r]*Σx*ιXLcheckexpanded[c]')+μx'*ιXLcheckexpanded[r]'*ιXLcheckexpanded[c]*μx
  end
  @assert all(moment(:EXLtXL; μx,Σx, dims) .≈ moment!(Val{:EXLtXL}(); μx,Σx,EXLtXL=rand(P+Δt,P+Δt), dims) .≈ EXLtXLcheck)
  EX̃LtX̃Lcheck = R'*EXLtXLcheck*R

  @assert all(moment(:EX̃LtX̃L; EXLtXL=moment(:EXLtXL; μx,Σx, dims), dims) .≈ EX̃LtX̃Lcheck)

  #@assert all(moment(:EX̃LtX̃L; μx,Λx, dims) .≈ moment!(Val{:EX̃LtX̃L}(); μx,Λx,EX̃LtX̃L=rand(P,P), dims) .≈ EX̃LtX̃Lcheck)

  @info "check on testformEXLtXL successful for (S,T,P,Δt)= $((S,T,P,Δt)) complete"

  return

end

function testmoment(V::Val{:EϕtX̃LtX̃Lϕ}; μϕ, Λϕ, μx, Λx, dims)
  Σx = Λx |> pdinv
  Σϕ = Λϕ |> pdinv

  EXLtXL = moment(Val{:EXLtXL}(); μx, Σx, dims)
 
  testmoment(Val{:EX̃LtX̃L}(); μx, Λx, dims)
  EX̃LtX̃L = moment(Val{:EX̃LtX̃L}(); EXLtXL, dims)
  
  @assert tr(EX̃LtX̃L*pdinv(Λϕ)) + μϕ'*EX̃LtX̃L*μϕ ≈ moment(V;EX̃LtX̃L, μϕ, Σϕ)

  return nothing
end


function testmoment(::Val{:EX̃Ltỹ}; 
    μx::AbstractVector,
    Λx=nothing, 
    Σx::AbstractMatrix=pdinv(Λx), 
    y,
    dims)
  @unpack S,T,P, ιXL, Δt, R =dims

  EXLtXL = Matrix{Float64}(undef, P+Δt,P+Δt)

  for r ∈ 1:(P+Δt), c ∈ 1:(P+Δt)
    EXLtXL[r,c] = tr(ιXL[r]*Σx*ιXL[c]') + μx'ιXL[r]'*ιXL[c]*μx
  end

  @assert all(EXLtXL .≈ moment(Val{:EXLtXL}(); μx, Σx, dims))

  EXL = moment(Val{:EXL}(); μx, dims)
  EX̃L =  EXL*dims.R
  @assert EX̃L ≈ moment(Val{:EX̃L}(); EXL, dims)

  ExS = moment(Val{:ExS}(); μx, dims)

  EX̃LtxS = dims.R'*([sum(tr(ιXL[i]*Σx*ιXL[j]') for j ∈ (P+1):(P+Δt)) for i in 1:(P+Δt)] + EXL'*ExS)  
  @assert EX̃LtxS ≈ moment(Val{:EX̃LtxS}(); EXLtXL, dims)


  @assert EX̃L'*y .- EX̃LtxS ≈ moment(Val{:EX̃Ltỹ}(); EXLtXL, EXL, y, dims)

  return
end

function testmoment(V::Val{:ExtΨx}; μx, Λx, EΨ)
  Σx = Λx |> inv

  @assert tr(EΨ*pdinv(Λx)) + μx'*EΨ*μx ≈ moment(V; μx, Σx, EΨ)

  return nothing
end


function testmoment(V::Val{:Eỹtỹ}; μx, Λx, y, dims)
  @unpack R, ιXL, Δt, P = dims
  Σx = Λx |> pdinv

  EXLtXL = moment(Val{:EXLtXL}(); μx, Σx, dims)


  ExS = moment(Val{:ExS}(); μx, dims)

  EXL = moment(Val{:EXL}(); dims, μx)
  #println(sum(tr(ιXL[2]*Σx*ιXL[j]')+ EXL'*ExS for j ∈ (P+1):(P+Δt)))
  EX̃LtxS = R'*([sum(tr(ιXL[i]*Σx*ιXL[j]') for j ∈ (P+1):(P+Δt)) for i in 1:(P+Δt)] + EXL'*ExS)
  @info EX̃LtxS 
  @info moment(Val{:EX̃LtxS}(); EXLtXL, dims)
  @assert all(EX̃LtxS .≈ moment(Val{:EX̃LtxS}(); EXLtXL, dims))
  ExStxS::Float64 = sum(tr(ιXL[i]*Σx*ιXL[j]')+μx'ιXL[i]'*ιXL[j]*μx for i ∈ (P+1):(P+Δt), j ∈ (P+1):(P+Δt))
  @assert ExStxS ≈ sum(EXLtXL[(P+1):end, (P+1):end])

  @assert y'y - 2y'formxS(μx; dims) +  ExStxS ≈ moment(V;  EXLtXL, y, μx, dims)

  return nothing
end

function testmoment(V::Val{:EβtFtPrtΨFβPr}; μβ, Λβ, EΨ, F, r)
  Σβ = Λβ |> inv

  #@assert moment(V; μβ, Λβ, F, EΨ) ≈ moment(V; μβ, Σβ, F, EΨ)
  @assert tr(F'*EΨ*F*pdinv(Λβ)) + μβ'*F'*EΨ*F*μβ + r'EΨ*F*μβ+μβ'*F'*EΨ*r + r'EΨ*r ≈ moment(V; μβ, Σβ, F, EΨ, r)

  return nothing
end



function testmoment(V::Val{:EβtFtPrT12tΨFβPrT12}; μβ, Λβ, EΨ, F, r,dims)
  @assert moment(V; μβ, Σβ=Λβ|>inv, F, EΨ, r,dims) ≈ moment(Val{:EβtFtPrtΨFβPr}(); μβ, Σβ=Λβ|>inv, EΨ, F, r=r*sqrt(dims.T))

  return nothing
end

function testmoment(V::Val{:EϕtM0ϕ}; μϕ, Λϕ, M0)
  Σϕ = Λϕ |> inv

  @assert tr(M0*pdinv(Λϕ)) + μϕ'*M0*μϕ ≈ moment(V; μϕ, Σϕ, M0)

  return nothing
end


function testmoment(::Val{:EDA0D}; A0, pγ, v, K=length(pγ))


  Eddt = Matrix{Float64}(undef, K, K)
  for r ∈ 1:K, c ∈ 1:K
    if r==c
      Eddt[r,r] = pγ[r]+(1.0-pγ[r])/v^2
    elseif r<c
      Eddt[r,c] = pγ[r]*pγ[c]+pγ[r]*(1-pγ[c])/v + (1-pγ[r])*pγ[c]/v + (1-pγ[r])*(1-pγ[c])/v^2
    else
      @assert r > c
    end
  end

  EDA0Dcheck = Symmetric(Eddt) .* A0
  ED = (pγ .+ (1.0 .- pγ)./v) |> Diagonal
  @assert ED.diag ≈ moment(:ED; pγ, v,).diag
  @assert EDA0Dcheck ≈ moment(:EDA0D; A0, ED, pγ, v, )
  @assert EDA0Dcheck ≈ moment(:EDA0D; A0, ED, pγ, v, EDA0D=rand(K,K))
end



function testmoment(V::Val{:ED}; pγ, v)
  K=length(pγ)

  @assert all((Diagonal(pγ .* formd.(trues(K),v) .+ (1 .- pγ)  .* formd.(falses(K),v)
    )) .≈ moment(V; pγ, v))

end

function testmoment(V::Val{:EβtDA0Dβ}; μβ, Λβ, EDA0D)
  Σβ = Λβ |> inv

  @assert tr(EDA0D*pdinv(Λβ)) + μβ'*EDA0D*μβ ≈ moment(V; μβ, Σβ, EDA0D)

  return nothing
end

function testmoment(V::Val{:Eβ̃tDA0Dβ̃}; μβ, Λβ, β0, EDA0D)
  Σβ = Λβ |> inv
  Eβ̃ = μβ-β0

  @assert tr(EDA0D*pdinv(Λβ)) + μβ'*EDA0D*μβ + β0'*EDA0D*β0 - 2*μβ'*EDA0D*β0  ≈ moment(V; Eβ̃, Σβ, EDA0D)

  return nothing
end



testmoment(s::Symbol, args...;kwargs...) = testmoment(Val{s}(),args...;kwargs...)


#the below work a little differently as they are meant to be called within the q update function
function testconditionalγkmoments(;
    dgp, k, EDGdk1,EDGdkv, EDA0DGdk1, EDA0DGdkv, Eβ̃tDA0Dβ̃Gdk1, Eβ̃tDA0Dβ̃Gdkv, Eβ̃tDA0βΔ0Gdk1,Eβ̃tDA0βΔ0Gdkv)
  @unpack Θ, hyper, dims = dgp
  @unpack K = dims
  @unpack β0, A0,v, βΔ0 = hyper
  @unpack ED, EDA0D, Σβ, μβ, pγ = dgp.Θ

  Eβ2 = diag(Σβ) .+ μβ.^2
 
  for dk ∈ [1.0,1/v]

    ########### Check EDGdk and EDA0DGdk
    EDGdk = deepcopy(ED)
    EDGdk[k,k] = dk
    @assert EDGdk ≈ (dk ≈ 1.0 ? EDGdk1 : EDGdkv)
    
    Eddt = Matrix(undef, K, K)
    for r ∈ 1:K, c ∈ 1:K
      if r==c
        Eddt[r,r] = r≠k ? pγ[r]+(1.0-pγ[r])/v^2 : dk^2
      elseif r<c
        if r==k
          Eddt[r,c] = dk*(pγ[c]+(1.0-pγ[c])/v)      
        elseif c==k
          Eddt[r,c] = dk*(pγ[r]+(1.0-pγ[r])/v)
        else
          @assert r ≠ k ≠ c
          Eddt[r,c] = pγ[r]*pγ[c]+pγ[r]*(1-pγ[c])/v + (1-pγ[r])*pγ[c]/v + (1-pγ[r])*(1-pγ[c])/v^2
        end
      else
        @assert r > c
      end
    end
    for r ∈ 1:K, c ∈ 1:K
      (r ≤ c) && continue
      Eddt[r,c] = Eddt[c,r]
    end

    EDA0DGdkcheck = Eddt .* A0
    @assert EDA0DGdkcheck ≈ (dk ≈ 1.0 ? EDA0DGdk1 : EDA0DGdkv)


    ######check EβtDA0β0Gdk and EβtDA0β0Gdk
    EDMkMk = view(ED, Not(k), Not(k))
    A0MkMk = view(A0, Not(k), Not(k))
    A0Mkk = view(A0, Not(k),k)
    μβMk = view(μβ,Not(k))
    β0Mk = view(β0, Not(k))
    βΔ0Mk = view(βΔ0, Not(k))
    a0kk = A0[k,k]
    ΣβMkMk = Σβ[Not(k),Not(k)]
    EDMkMkA0DMkMk = EDMkMk*A0MkMk*EDMkMk

    for (r,k) ∈ enumerate(collect(1:K)[Not(k)])
      EDMkMkA0DMkMk[r,r] = (pγ[k] .+ (1-pγ[k])/v^2)*A0[k,k]
    end

    @assert EDMkMkA0DMkMk ≈ EDA0D[Not(k), Not(k)]
    @assert μβMk'*EDMkMkA0DMkMk*μβMk .+ tr(EDMkMkA0DMkMk*Σβ[Not(k),Not(k)]) ≈ moment(
        Val{:EβtDA0Dβ}(); EDA0D=EDMkMkA0DMkMk, μβ=μβMk,Σβ=ΣβMkMk)
    
    #make sure that we have set this up as expected
    EDA0Ddkcheck2 = similar(EDA0D)
    EDA0Ddkcheck2[Not(k),Not(k)] .= EDMkMkA0DMkMk |> deepcopy
    EDA0Ddkcheck2[k,Not(k)] .= vec(dk .* A0Mkk'*EDMkMk)
    EDA0Ddkcheck2[Not(k),k] .= vec(dk*A0Mkk'*EDMkMk)
    EDA0Ddkcheck2[k,k] = a0kk*dk^2 
    @assert EDA0Ddkcheck2 ≈ (dk ≈ 1.0 ? EDA0DGdk1 : EDA0DGdkv)
    


    EβtDA0DβGdkcheck = (Eβ2[k]*a0kk*dk^2 
      + 2μβ[k]*dk*A0Mkk'*EDMkMk*μβMk + 2*dk*A0Mkk'*EDMkMk*Σβ[Not(k),k]
      + μβMk'*EDMkMkA0DMkMk*μβMk + tr(EDMkMkA0DMkMk*ΣβMkMk))

    Eβ0tDA0Dβ0Gdkcheck = (β0[k]^2*a0kk*dk^2 + 2β0[k]*dk*A0Mkk'*EDMkMk*β0Mk + β0Mk'*EDMkMkA0DMkMk*β0Mk) 
    EβtDA0Dβ0Gdkcheck = (β0[k]*μβ[k]*a0kk*dk^2 + μβ[k]*dk*A0Mkk'*EDMkMk*β0Mk + β0[k]*dk*A0Mkk'*EDMkMk*μβMk + μβMk'*EDMkMkA0DMkMk*β0Mk)   
    Eβ̃tDA0Dβ̃Gdkcheck = EβtDA0DβGdkcheck -2*EβtDA0Dβ0Gdkcheck+Eβ0tDA0Dβ0Gdkcheck
    #Aside- the second line of the above is equiv to the standard formula where β is augmented by a selection matrix   
    #The below verifies this
    ιkk = Diagonal(zeros(K))
    ιkk[k,k] = 1.0
    ιMkMk = Diagonal(ones(K))
    ιMkMk[k,k] =  0.0
    @assert ((μβ[k]*dk*A0Mkk'*EDMkMk*μβMk + dk*A0Mkk'*EDMkMk*Σβ[Not(k),k]) ≈ 
      (μβ'ιkk*dk*A0*ED*ιMkMk*μβ + tr(ιkk*dk*A0*ED*ιMkMk*Σβ)))

    #main check of βtDA0DβGdk
    @assert Eβ̃tDA0Dβ̃Gdkcheck ≈ (dk ≈ 1.0 ? Eβ̃tDA0Dβ̃Gdk1 : Eβ̃tDA0Dβ̃Gdkv) "
    Eβ̃tDA0Dβ̃Gdkcheck=$Eβ̃tDA0Dβ̃Gdkcheck, $((dk ≈ 1.0 ? "Eβ̃tDA0Dβ̃Gdk1: $Eβ̃tDA0Dβ̃Gdk1"  : "EβtDA0DβGdkv: $Eβ̃tDA0Dβ̃Gdkv"))"


    EβtDA0βΔ0Gdkcheck = μβ[k]*dk*A0[k,:]'*βΔ0 + μβMk'*EDMkMk*A0[Not(k),:]*βΔ0
    Eβ0tDA0βΔ0Gdkcheck = β0[k]*dk*A0[k,:]'*βΔ0 + β0Mk'*EDMkMk*A0[Not(k),:]*βΔ0
    Eβ̃tDA0βΔ0Gdkcheck = EβtDA0βΔ0Gdkcheck - Eβ0tDA0βΔ0Gdkcheck
    @assert Eβ̃tDA0βΔ0Gdkcheck ≈ (dk ≈ 1.0 ? Eβ̃tDA0βΔ0Gdk1 : Eβ̃tDA0βΔ0Gdkv)


    
  end

end


function testEν(dgp; iters=[10^i for i ∈3:6], gausslimit = 10^4, 
    runprecisiontests=true, runtimingtests=true)

  @unpack lpν = updateθpν(dgp)
  @unpack νmin, νmax = dgp.hyper
  pν(ν) = exp(lpν(ν))

  @info "Reference: default adaptive"
  Eνest = moment(Val{:Eν}(), pν; νmin, νmax)
  println("Eν (adaptive): $Eνest")

  if runprecisiontests && isfinite(νmax-νmin)
    for iter ∈ iters
      if iter > gausslimit
        @info "iter $iter > gausslimit $gausslimit. Skipping precision test- increase limit if test is still desired"
        continue
      end

      nodes = computeνnodes(gauss, Float64, iter; νmin, νmax)
      (νg, wg) = gauss(iter, νmin, νmax)
      @assert nodes.ν ≈ νg
      @assert nodes.w ≈ wg

      νmingauss = minimum(nodes.ν)
      νmaxgauss = maximum(nodes.ν)

      Eνest = moment(Val{:Eν}(), pν, nodes;)
      println("Eν: $Eνest (gauss, iter=$iter, νmin=$νmingauss, νmax=$νmaxgauss)")
    end

    print("\n")
    for iter ∈ iters
      nodes = computeνnodes(gausschebyshev, iter, 1; νmin, νmax)
      Eνest = moment(Val{:Eν}(), pν, nodes;)
      println("Eν: $Eνest (gausschebyshev (1), iter=$iter)")
    end

    print("\n")
    for iter ∈ iters
      nodes = computeνnodes(gausschebyshev, iter, 2; νmin, νmax)
      Eνest = moment(Val{:Eν}(), pν, nodes;)
      println("Eν: $Eνest (gausschebyshev (2), iter=$iter)")
    end

    print("\n")
    for iter ∈ iters
      nodes = computeνnodes(gausslegendre, iter; νmin, νmax)
      Eνest = moment(Val{:Eν}(), pν, nodes;)
      println("Eν: $Eνest (gausslegendre, iter=$iter)")
    end

    print("\n")
    for iter ∈ iters
      nodes = computeνnodes(gausslobatto, iter; νmin, νmax)
      Eνest = moment(Val{:Eν}(), pν, nodes;)
      println("Eν: $Eνest (gausslobatto, iter=$iter)")
    end 
  end 

  if runprecisiontests
    print("\n")
    for iter ∈ iters
      nodes = computeνnodes(gausslaguerre, iter,; νmin, νmax, rescale=false)
      Eνest = moment(Val{:Eν}(), pν, nodes;)
      println("Eν: $Eνest (gausslaguerre (bounds of 0,∞), iter=$iter)")
    end 
  end

  if runtimingtests
    print("\nTiming adaptive baseline: ")
    @btime moment(Val{:Eν}(), $pν; νmin=$νmin, νmax=$νmax)

    for iter ∈ iters
      nodes = (isfinite(νmax-νmin) ? 
        computeνnodes(gausslegendre, iter; νmin, νmax) :
        computeνnodes(gausslaguerre, iter,; νmin, νmax, rescale=false))
      print("Timing computation|nodes ($iter): ")
      @btime moment(Val{:Eν}(), $pν, $nodes;)
    end
  end

end