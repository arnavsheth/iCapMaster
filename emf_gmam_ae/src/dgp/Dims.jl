######################################
#immutable dimensional information
struct Dims{Ts2t,Ts2allts,TιXL,Tιϕ,TR, Tdates, TFθ2name, Tpriorset} <: AbstractDims
    T::Int
    S::Int
    P::Int
    K::Int
  
    Δt::Int
    #s2Lts::Ts2Lts #maps s to the lagged window
    s2t::Ts2t #maps s to t
    s2allts::Ts2allts #maps s to both the lag window and the contemporaneous t
    ιXL::TιXL #selection matrix for lagging x
    ιϕ::Tιϕ #selection vectors for restrictions on
    R::TR #restriction matrix for phi [I(P); ιϕ[1]'; ..; ιϕ[Δt]']

    dates::Tdates
    Fθ2name::TFθ2name
    priorset::Tpriorset

  end

#basically a dims recycling function
function Dims(dims::AbstractDims;
  K=dims.K, S=dims.S, P=dims.P, Δt=dims.Δt, 
  dates=dims.dates, 
  Fθ2name = dims.Fθ2name,
  priorset=dims.priorset,
  addxylabels=true)

  dims = Dims(;S, P,K, Δt, dates, Fθ2name, priorset, addxylabels)
end   
#this generates a function that maps a data index s to the index values used for the factors and latent variables t
#e.g., with monthly data for y and monthly for F and a lookback of 6 mo, s=1 coresponds to 1:6 and s=3 corresponds to 3:9
#e.g., with quarterly data for y and monthly for F and a lookback of 12 mo, s=1 coresponds to 1:12 and s=3 corresponds to 7:19
#generates2Lts(;Δt::Int,P::Int,S::Int) = [((s-1)*Δt+1):(P + (s-1)*Δt) for s ∈ 1:S]
#generates2allts(;Δt::Int,P::Int,S::Int) = generates2Lts(;Δt,P,S) .|> r->minimum(r):(maximum(r)+Δt)
generates2allts(;Δt::Int,P::Int,S::Int) = [((s-1)*Δt+1):(P + Δt + (s-1)*Δt) for s ∈ 1:S]
  
#form the dimensions and the auxiliarry mappings
function Dims(; K, S, P,Δt, 
    dates = nothing, 
    Fβs=nothing, 
    testdims=false,
    tlocalparams=PARAM[:iotlocalparams],
    slocalparams = PARAM[:ioslocalparams],
    addxylabels=true,
    Fθ2name::TFθ2name = Dict{Symbol, Any}(),
    priorset="none") where TFθ2name
  


  #s2Lts = generates2Lts(;Δt, P, S)
  s2allts = generates2allts(;Δt, P, S)

  if !isempty(s2allts)
    s2t = (s2allts .|> maximum)
    T = s2t[end]
    if dates !== nothing
      @assert T == length(dates)
      @assert issorted(dates)
    end
  else #the empty priors only case
    s2t = Float64[]
    T=0
    @assert S==length(something(dates, Date[]))
  end

  #these labels can be helpful in interpreting the results
  if (dates !== nothing) && (TFθ2name !== nothing)
    for Fθ ∈ [tlocalparams; addxylabels ? [:x;] : Symbol[]]
      Fθ2name = merge(Fθ2name, Dict(Symbol("$Fθ[$t]")=>dates[t] for t ∈ 1:T))
    end

    for Fŷ ∈ [slocalparams; addxylabels ? [:y;] : Symbol[]]
      Fθ2name = merge(Fθ2name, Dict(Symbol("$(Fŷ)[$s]")=>dates[s2t[s]] for s ∈ 1:S))
    end
  end

  if Fβs !==nothing && (TFθ2name !== nothing)
    Fθ2name = merge(Fθ2name, Dict(Symbol("β[$k]")=>Fβs[k] for k ∈ 1:K))
    Fθ2name = merge(Fθ2name, Dict(Symbol("γ[$k]")=>Fβs[k] for k ∈ 1:K))
  end

  ιXL = formιXL(;P,T,S,s2t, Δt)
  ιϕ = formιϕ(; P, Δt)
  Rtop = I(P)
  Rbottom = (-1 .* ιϕ) |> transpose
  R = vcat(Rtop, Rbottom)
  @assert size(R) ≡ (P+Δt,P)


  if testdims
    tests2t(; )
    testιXL(; )
    testιϕ(; Δt, P, R)
  end



  dims = Dims(T,S, P,K, Δt, s2t, s2allts, ιXL,ιϕ, R , dates, Fθ2name, priorset)


  return dims
end

function addθnames!(dims::Dims; Fθ2name)
  #we want to be careful not to overwrite
  keys2add = setdiff(Fθ2name |> keys, dims.Fθ2name |> keys)
  @assert isempty(intersect(dims.Fθ2name |> keys, keys2add))

  merge!(dims.Fθ2name, Dict(k .=> Fθ2name[k] for k ∈ keys2add))

  return nothing

end

function truncate(old::Dims; smax)
  @unpack K,Δt, P, s2t = old
  S = smax
  dims=Dims(;K,Δt, P, S,)

  @assert dims.T ≈ s2t[smax]
  return dims
end


function tests2t(; )
  S=100
  K=4
  Δt=1
  P=6


  dims=Dims(;S,K,P,Δt,)
  @unpack #=s2Lts, =#s2allts, s2t = dims
  #@assert 1:6 ≡ s2Lts[1]
  @assert 1:7 ≡ s2allts[1]  
  #@assert 3:8 ≡ s2Lts[3]
  @assert 3:9 ≡ s2allts[3]  
  #@assert 100:105 ≡ s2Lts[S] ≡ s2Lts[end]
  @assert 100:106 ≡ s2allts[S] ≡ s2allts[end]
  @assert all((s2allts .|> maximum) .== s2t)

  Δt=3
  P=12
  dims=Dims(;S,K,P,Δt,)
  @unpack #=s2Lts, =#s2allts, s2t = dims
  #@assert 1:14 ≡ s2Lts[1]
  @assert 1:15 ≡ s2allts[1]
  #@assert 7:20 ≡ s2Lts[3]
  @assert 7:21 ≡ s2allts[3]
  #@assert 298:311 ≡ s2Lts[S] ≡ s2Lts[end]
  @assert 298:312 ≡ s2allts[S] ≡ s2allts[end]
  @assert all((s2allts .|> maximum) .== s2t)

end


#=function formιXL(; P, T, S, s2t, Δt)

  #These matrices map x to the columns of XL e.g. ιXL[:,:,p]*x=XL[:,p]
  #They effectively lag the values of x for the observation periods
  #Another way to look at it: the rule is x(i) is selected at row (j) if:
  # 1) x(i) is present in column p of XL
  # 2) t(s) == row j, where s is the location of x(i) in column p of XL
  # A third way of looking at it: ιXL[i,j,p]==true IFF 
  #  There exists an s s.t. x[j]==XL[s,p] and t(s)==i

  ιXLdense = [falses(T,T) for p ∈ 1:(P+Δt)]
  for p ∈ 1:(P+Δt)
    #@info "p=$p, s2t=$s2t"
    selections = CartesianIndex.(s2t, s2t .- P .- Δt .+ p)
    ιXLdense[p][selections] .= true
  end

  ιXL = ιXLdense .|> sparse

  #sanity check
  @assert sum(sum(ιXL)) == (P+Δt) * length(s2t)

  return ιXL
end=#

function formιXL(; P, T, S, s2t, Δt)


  #These matrices map x to the columns of XL e.g. ιXL[:,:,p]*x=XL[:,p]
  #They effectively lag the values of x for the observation periods
  #Another way to look at it: the rule is x(i) is selected at row (j) if:
  # 1) x(i) is present in column p of XL
  # 2) t(s) == row j, where s is the location of x(i) in column p of XL
  # A third way of looking at it: ιXL[i,j,p]==true IFF 
  #  There exists an s s.t. x[j]==XL[s,p] and t(s)==i

  ιXLdense = [falses(S,T) for p ∈ 1:(P+Δt)]

  for p ∈ 1:(P+Δt)
    #@info "p=$p, s2t=$s2t"
    S==0 && continue
    
    selections = CartesianIndex.(1:S, s2t .- P .- Δt .+ p)
    ιXLdense[p][selections] .= true

  end

  ιXL = ιXLdense .|> sparse

  #sanity check
  @assert sum(sum(ιXL)) == (P+Δt) * length(s2t)

  return ιXL
end



function testιXL(;)
  #form the test data
  S = 100

  #simple case
  Δt=1
  P=6
  #s2Lts= generates2Lts(; Δt, P, S)
  s2t = [(Δt*(s-1) + P + Δt) for s ∈ 1:S]
  s2allts = ((l,h)->l:h).(s2t .- P .- Δt .+ 1,s2t)
  T = maximum(s2allts[end])
  ιXL = formιXL(; P, T, S, s2t, Δt)
  x = rand(T)
  dims=(; T,S,P, Δt, s2t, s2allts, ιXL)

  #note XL_alt generates the XL matrix from the sparse matrices
  XL = _formXL(x; dims)
  XLalt = _formXL_alt(x; dims)
  @assert all((XL .≈ XLalt) .| ((XL .+ 1.0) .≈ (XLalt .+ 1.0)))

  #quarterly case
  Δt=3
  P=12
  #s2Lts= generates2Lts(; Δt, P, S)
  s2t = [(Δt*(s-1) + P + Δt) for s ∈ 1:S]
  s2allts = ((l,h)->l:h).(s2t .- P .- Δt .+ 1,s2t)
  T = maximum(s2allts[end])
  ιXL = formιXL(; P, T, S, s2t, Δt)
  x = rand(T)
  dims=(; T,S,P, Δt, s2t, s2allts, ιXL)

  #note XL_alt generates the XL matrix from the sparse matrices
  XL = _formXL(x;dims)
  XLalt = _formXL_alt(x; dims)
  @assert all((XL .≈ XLalt) .| ((XL .+ 1.0) .≈ (XLalt .+ 1.0)))


end

function formιϕ(;P,Δt)
  @assert P % Δt == 0

  ιϕ = [((p-l)%Δt) == 0 for p ∈ 1:P, l ∈ 1:Δt]

  @assert size(ιϕ) ≡ (P,Δt)

  return ιϕ
end


function testιϕ(;P, Δt, R)
  ιϕact = formιϕ(;P,Δt)
  Rbottom = R[(P+1):end, :]
  @assert all([(((p-l) % Δt)==0) for p ∈ 1:P, l ∈ 1:Δt]' .== ιϕact' .== (-Rbottom))

  ϕ = rand(P)
  ϕ̃ = [ϕ; [1.0 - sum(ϕ[l:Δt:P]) for l ∈ 1:Δt];]
  @eval Main ϕ̃=$ϕ̃
  @assert  formϕ̃(; ϕ, dims=(;Δt, P, ιϕ=ιϕact)) ≈ R*ϕ .+ [zeros(P); ones(Δt)]
  
  ϕ̃act = R*ϕ .+ [zeros(P); ones(Δt)]
  @eval Main ϕ̃act = $ϕ̃act
  @assert ϕ̃ ≈ ϕ̃act
   
  @assert all(formιϕ(;P=6,Δt= 3)' .== (
    [1 0 0 1 0 0;
     0 1 0 0 1 0;
     0 0 1 0 0 1;]
  ))


  return nothing
end
