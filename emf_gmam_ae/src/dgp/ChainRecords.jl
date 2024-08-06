

#this basically is an access wrapper around MCMCChains
@kwdef struct ChainRecords{Tchainid<:Union{Symbol, AbstractString}, 
    Tchain, 
    Tchainparts,
    Tfieldstocapture<:AbstractVector{Symbol},
    Tchainfieldindex,
    Tnumburnrecords}

  chainid::Tchainid

  chain::Tchain
  chainparts::Tchainparts
  fieldstocapture::Tfieldstocapture
  chainfieldindex::Tchainfieldindex

  lastrecord::Vector{Int}
  burnt::Bool

  numrecords::Int
  numsamplerecords::Int
  numburnrecords::Tnumburnrecords
  numchains::Int
end

Base.getproperty(rec::Trec, p::Symbol) where Trec <: ChainRecords = hasfield(Trec, p) ? getfield(rec, p) : (
  haskey(rec.chainparts,p) ? rec.chainparts[p] : rec.chain[:,p,:])

#more helper methods for getting properties, this time via getindex
Base.getindex(records::Trecords, p::Symbol) where Trecords <: ChainRecords = getproperty(records,p)
Base.getindex(records::Trecords, ps::AbstractVector{Symbol}) where Trecords <: ChainRecords = (
    any(ps .|> p->hasfield(Trec,p)) ? (p->records[p]).(ps) : chain[:,expandchainfields(rec, ps),:])
Base.getindex(records::Trecords, args...) where Trecords <: ChainRecords = records[args]

#tehse methods create data structures for the records
initializerecordpart(::T;numrecords,numchains) where T<: Real = missings(T, numrecords,1,numchains)
initializerecordpart(v::AbstractVector{T};numrecords, numchains) where T<: Real = missings(T, numrecords, length(v),numchains)
initializerecordpart(v::AbstractArray{T};numrecords, numchains) where T<: Real = missings(T, numrecords, length(v),numchains)
initializerecordpart(::T;kwargs...)  where T = throw("Unrecognized parameter type $T")

expandchainfield(rec::Trec, p::Symbol) where Trec <: ChainRecords = haskey(rec.chainfieldindex, p) ? rec.chainfieldindex[p] : p
expandchainfields(rec::Trec, ps::Vector{Symbol}) where Trec <: ChainRecords = reduce(vcat, [expandchainfield(rec,p) for p ∈ ps])

function Base.empty(records::ChainRecords; 
    dgp=nothing,
    Θ = dgp.Θ,
    numburnrecords::Int=records.numburnrecords,
    numsamplerecords::Int=records.numsamplerecords,
    removeaugmentations=true,
    fieldstocapture = (removeaugmentations ?
      intersect(records.fieldstocapture, propertynames(Θ)) :
      records.fieldtocapture |> deepcopy))

  @unpack numchains, chainid =records

  @assert removeaugmentations && isempty(setdiff(fieldstocapture, propertynames(Θ))) ".
    removeaugmentations=$removeaugmentations but fieldstocapture=$fieldstocapture and 
    propertynames(Θ)=$(propertynames(Θ))"


  emptied = ChainRecords(fieldstocapture; Θ, chainid, numsamplerecords, numburnrecords, numchains,)
  return emptied
end

Base.isempty(records::ChainRecords; Θ=nothing) = all(records.lastrecord .== 0)

function ChainRecords(fieldstocapture::AbstractVector; Θ=nothing, chainid,
    testchainrecords=false,
    numsamplerecords,
    numburnrecords=nothing,
    numchains,
    numrecords = numsamplerecords + something(numburnrecords,0),
    
    #create a template data object
    inits=[f=>
      initializerecordpart(getproperty(Θ,f); numrecords, numchains) for f ∈ fieldstocapture],
    burnt::Bool=(numburnrecords ≡ nothing),
    )

  chns = Vector()

  #check dimensional consistency
  @assert burnt == (numburnrecords≡nothing)
  @assert something(numburnrecords,0)+numsamplerecords == numrecords
  

  #create the chain fragments
  fnames = Dict{Symbol, Any}()
  for (f,obj) ∈ inits
    fnames[f] = (size(obj,2) > 1 ? ["$f[$i]" for i ∈ 1:size(obj,2)] : ["$f"]) .|> Symbol
    push!(chns, Chains(obj, fnames[f]))
  end

  chain = reduce(hcat, chns)

  #@eval Main chain=$chain

  #the below indexes a view into each property in the chain
  #chainparts = Dict(f=>
  #  length(fnames[f]) == 1 ? vec(@view(chain.value[:,fnames[f],:])) : @view(chain.value[:,fnames[f],:]) for  f ∈ fieldstocapture)
  chainparts = Dict(f=>@view(chain.value[:,fnames[f],:]) for  f ∈ fieldstocapture)
  fieldstocapture = fieldstocapture .|> Symbol
  lastrecord = zeros(Int, numchains)
  chainfieldindex = (;OrderedDict(f=>namesingroup(chain, f) for  f ∈ fieldstocapture)...)
  records = ChainRecords(;chainid, chain, chainparts,fieldstocapture,chainfieldindex, lastrecord, burnt,
    numrecords, numsamplerecords, numburnrecords, numchains,)

  testchainrecords && validatechainrecords(records; Θ,)

  return records
end

augmentchainrecords(records::ChainRecords; augmentation, Fgroup) = augmentchainrecords(records, Dict(Fgroup=>augmentation))
function augmentchainrecords(records::ChainRecords, augmentations::AbstractDict; )
  @unpack chainid, lastrecord, numrecords, numburnrecords, numsamplerecords, numchains, burnt, chainparts  = deepcopy(records)


  #Nnew = size(augmentation,2)
  #@assert size(augmentation) ≡ (numrecords, Nnew, numchains)
  #do not want to overwrite
  @assert intersect(records.fieldstocapture, augmentations |> keys |> collect) |> isempty
  fieldstocapture::Vector{Symbol} = [records.fieldstocapture; augmentations |> keys |> collect]


  


  #create the chain fragments
  #=Fnames = map(keys(augmentations)) do Fgroup
    (Nnew > 1 ? ["$Fgroup[$i]" for i ∈ 1:Nnew] : ["$Fgroup"]) .|> Symbol
  end |> nms->reduce(vcat, nms)=#

  function genFnames(Fgroup)
    Nnew = size(augmentations[Fgroup],2)
    (Nnew > 1 ? ["$Fgroup[$i]" for i ∈ 1:Nnew] : ["$Fgroup"]) .|> Symbol
  end

  Fnames = OrderedDict(k=>genFnames(k) for k ∈ keys(augmentations))

  #Fnames = (Nnew > 1 ? ["$Fgroup[$i]" for i ∈ 1:Nnew] : ["$Fgroup"]) .|> Symbol

  chainaugmentations = map(keys(augmentations) |> collect) do Fgroup

    @assert size(augmentations[Fgroup],1) == (burnt ? numsamplerecords : numrecords)

    return Chains(augmentations[Fgroup], Fnames[Fgroup])
  end

  #chainaugmentations = [Chains(augmentations[Fgroup], Fnames[Fgroup]) for Fgroup ∈ keys(augmentations)]
  #@info size(records.chain)
  #@eval Main records=$records
  #@eval Main augmentations=$augmentations
  #@eval Main chainaugmentations=$chainaugmentations
  chain = hcat(records.chain, chainaugmentations...)

  chainfieldindex = records.chainfieldindex
  for Fgroup ∈ keys(augmentations)
    chainparts[Fgroup] = @view(chain.value[:,Fnames[Fgroup],:])
    chainfieldindex=merge(chainfieldindex, (;Dict(Fgroup=>namesingroup(chain, Fgroup))...))
  end

  augmented = ChainRecords(; 
    chainid,
    chain,
    fieldstocapture,
    lastrecord,
    chainparts,
    numrecords,
    numburnrecords, 
    numsamplerecords,
    numchains,
    chainfieldindex, 
    burnt)

  return augmented
end



function record!(dgp; chainnum=1, kwargs...)
  @unpack records, Θ = dgp

  return record!(records; chainnum, Θ, kwargs...)
end

function record!(records::ChainRecords;chainnum = 1, Θ, debugnow=false)
  #check that we have a complete record to store and the space to store it
  @unpack numrecords = records
  @unpack fieldstocapture = records
  @assert records.lastrecord[chainnum] < numrecords

  #@assert isempty(setdiff(records.fieldstocapture,propertynames(Θ))) "records.fieldstocapture=$(records.fieldstocapture) 
  #  but propertynames(Θ)=$(propertynames(Θ))!"

  #store a pointer to the current record
  i = records.lastrecord[chainnum] + 1


  for f ∈ fieldstocapture
    try
      getproperty(records, f)[i,:,chainnum] .= vecmaybe(Θ[f])
    catch err
      @eval Main p = $(getproperty(records, f))

      @info "Failed to store in record number $i, chain number $chainnum, the value stored in key $f: val $(Θ[f])"
      throw(err)
    end
  end


  records.lastrecord[chainnum] = i

  return
end






function burnchainrecords(; dgp, )

  @unpack Θ,records,dims = dgp

  @unpack numburnrecords, numsamplerecords, numchains, fieldstocapture, chainparts, chainfieldindex  = records
  @unpack S = dims
  all(records.lastrecord .< numburnrecords) && throw("number of recorded records is less than burn-in") 

  initstoreduce=map(fieldstocapture) do Fθ
    if hasproperty(Θ,Fθ)
      return Fθ=>initializerecordpart(getproperty(Θ,Fθ); numrecords=numsamplerecords, numchains)
    else
      θeltype = chainparts[Fθ][begin] |> typeof
      Nstat = length(chainfieldindex[Fθ])
      return Fθ=>missings(θeltype, numsamplerecords, Nstat ,numchains)
    end
  end

  inits = reduce(vcat, initstoreduce)
  
  #@eval Main fieldstocapture=$fieldstocapture
  burnt = ChainRecords(fieldstocapture; Θ, records.chainid, numsamplerecords, numchains, 
    inits,
    burnt=true)

  
  for f ∈ burnt.fieldstocapture
    burnt[f] .= records[f][(numburnrecords+1):end, :, :]
  end

  burnt.lastrecord .= numsamplerecords
  @assert burnt.burnt

  return burnt

end

function expandchainrecords(;
    dgp=nothing, 
    Θ=dgp.Θ, 
    oldrecords=dgp.records,
    sampleexpandfrac,
    samplerecyclefrac,
    maxrecords=typemax(Int)
    )

    @assert samplerecyclefrac ≤ 1.0

    @unpack numsamplerecords, numrecords, numburnrecords = oldrecords

    #the total number of expanded records should be always less than or equal to max records
    targetnewrecords = round((numsamplerecords * (1+sampleexpandfrac - samplerecyclefrac)),RoundUp) |> Int
    nextnumrecords = min(numrecords +  targetnewrecords,  maxrecords)

      
    #the sample size must never go down (neither should the burn-in size)
    nextnumsamplerecords = max(numsamplerecords, nextnumrecords-numrecords + Int(round(samplerecyclefrac*numsamplerecords, RoundDown)))
    nextnumburnrecords = nextnumrecords - nextnumsamplerecords

    @assert numsamplerecords ≤ nextnumsamplerecords ≤ round(numsamplerecords * (1+sampleexpandfrac), RoundUp) "
      Failed assertion numsamplerecords ≤ nextnumsamplerecords ≤ numsamplerecords * (1+sampleexpandfrac)
      numsamplerecords=$numsamplerecords; nextnumsamplerecords=$nextnumsamplerecords; $(
          round(numsamplerecords * (1+sampleexpandfrac), RoundUp))"
    @assert (numsamplerecords==nextnumsamplerecords) || (
        Int(targetnewrecords + Int(round(samplerecyclefrac*numsamplerecords, RoundDown)) - nextnumsamplerecords) ≈ 
        targetnewrecords - (nextnumrecords-numrecords))

    return expandchainrecords(nextnumburnrecords, nextnumsamplerecords; Θ, oldrecords)
end

function expandchainrecords(numburnrecords::Int, numsamplerecords::Int; Θ, oldrecords) 


  @assert !oldrecords.burnt
  @assert oldrecords.numburnrecords ≤ numburnrecords
  @assert oldrecords.numsamplerecords ≤ numsamplerecords
  @assert allequal(oldrecords.lastrecord)

  records = ChainRecords(oldrecords.fieldstocapture; 
    Θ,
    chainid=oldrecords.chainid,
    numburnrecords, 
    numsamplerecords, 
    numchains=oldrecords.numchains, 
    burnt=false)

  for f ∈ records.fieldstocapture
    records[f][1:(oldrecords.numrecords),:,:] .= oldrecords[f]
  end

  records.lastrecord .= oldrecords.lastrecord

  return records

end




function loadrecords(; 
    analysispath = PARAM[:analysispath],
    rid=PARAM[:iorid],
    inbinstream=IN_BIN_STREAM )

  outputpath = "$analysispath/$rid"
  dgp = "$analysispath/$rid/chain/$rid.zstd" |> inbinstream


  return dgp
end

function validatechainrecords(recordsin::ChainRecords;  Θ,)
  records = recordsin |> deepcopy

  @assert isempty(setdiff(keys(records.chainparts) |> collect, records.fieldstocapture))
  #verify initialization
  @assert records.numrecords == records.numburnrecords + records.numsamplerecords
  @assert records.numrecords == size(records.chain.value,1)
  @assert records.numchains == size(records.chain.value,3)

  @eval Main records=$records

  for f ∈ records.fieldstocapture
    chainpart = records[f]
    @assert chainpart ≡ records.chainparts[f]
    chainpart .= 1
    
    chainprop = get(records.chain, f) |> nt-> !isempty(nt) ? nt[f] : records.chain[f] 
    if typeof(chainprop) <: AbstractMatrix #should be the case when the param is a scalar
      chainprop3d = [chainprop[i,c] for i in 1:records.numrecords, t in 1:1, c in 1:records.numchains]
    elseif typeof(chainprop) <: Tuple
      chainprop3d = [chainprop[t][i,c] for i in 1:records.numrecords, t in 1:length(chainprop), c in 1:records.numchains]
    else
      throw("unrecognized type of chainprop $(chainprop) for param $f")
    end

    #@info f
    

    #@eval Main chainprop=$chainprop
    #@eval Main chainpart=$chainpart
    @assert all(chainprop3d .== chainpart .== 1)

  end

  ######now verify recording
  #create some sample records
  numchains=3
  numrecords=1000
  numburnrecords=200
  numsamplerecords=800
  Θtest = (;a=10.0,b=rand(10), c=rand(10,10), d=9.2)
  #store the records
  fieldstocapture = [:b,:c,:d]
  recordstest = ChainRecords(fieldstocapture,; Θ=Θtest, chainid=:_, 
    testchainrecords=false,  burnt=false, numrecords, numchains, numburnrecords, numsamplerecords)

  Θs = [[(;a=rand(),b=rand(10), c=rand(10,10), d=rand()) for i ∈ 1:numrecords] for numchain ∈ 1:numchains]
  #@eval Main Θs=$Θs

  #need to reshape this for comparisons, as arrays of tuples are hard to work with
  #first, destructure the items within the tuples, and stack each field into a vector (potentially of other vectors)
  Θcollapsed = [
      [
        (;a=Θs[numchain][i].a,b=Θs[numchain][i].b, c=Θs[numchain][i].c |> vec, d=Θs[numchain][i].d) 
      for i ∈ 1:numrecords] 
    for numchain ∈ 1:numchains] .|> collapse

  #now, concatenate the vectors or numbers into matrices or vectors respectivelly
  Θvec = [
    (;
      Dict(n=>reduce(hcat, Θcollapsed[numchain][n])' 
      for n ∈ propertynames(Θtest))...) 
    for numchain ∈ 1:numchains] 


  @eval Main recordstest=$recordstest
  for r ∈ 1:numrecords, chainnum ∈ 1:numchains
    record!(recordstest; chainnum, Θ=Θs[chainnum][r])
  end

  #verify the records
  @eval Main recordstest = $recordstest
  @eval Main Θvec = $Θvec
  for chainnum ∈ 1:numchains, f ∈ fieldstocapture
    @assert all((10.0 .+ getproperty(Θvec[chainnum],f)) .≈ (recordstest[f][:,:,chainnum] .+ 10.0))
  end

  

  ######verify dgp recording mechanism
  #create a stub DGP
  dims=  m.Dims(; 
    K=PARAM[:simulateK], 
    S=PARAM[:simulateS], 
    P=PARAM[:simulateP],
    Δt=PARAM[:simulatefrequency], 
    )

  ####verify burn-in process
  burnt = burnchainrecords(;dgp=(;records=recordstest, Θ=Θtest,dims) )
  #@eval Main burnt=$burnt
  @assert burnt.burnt

  for f ∈ fieldstocapture
    @assert all(recordstest[f][(numburnrecords+1):end,:,:] .== burnt[f])
  end

  @unpack T,K,P,S = dims
  fieldstocapture=[:x,:ϕ,:β,:γ,:ω,:τy,:τx,:ψ,:ν]

  ϕtest = rand(P)
  ψtest = rand(T)
  νtest = rand()
  ωtest = rand()
  τxtest = rand()
  τytest = rand()
  γtest = rand([false,true],K)
  βtest = rand(K)
  xtest = rand(T)
  Θ = DGPModelParameters(;x=xtest, ϕ=ϕtest, β=βtest, γ=γtest, ω=ωtest, τy=τytest, τx=τxtest, ψ=ψtest, ν=νtest)
  hyper = Θ |> deepcopy
  records=ChainRecords(fieldstocapture, ; Θ, chainid=:_, testchainrecords=false, burnt=false, numburnrecords, numsamplerecords, numchains)
  data=Data(;y=rand(S), F=rand(T,K), r=rand(T))
  dgp = DGP(; hyper, Θ, dims, records, data)

  #record and verify
  [record!(dgp; chainnum=c, debugnow=true) for i ∈ 1:5, c ∈ 1:records.numchains]
  @assert records.lastrecord[1]==5
  @assert issetequal(fieldstocapture, records.fieldstocapture)

  for f ∈ fieldstocapture
    for c ∈ 1:records.numchains
      @assert all(records[f] .≡ records[f][:,records.chainfieldindex[f],:])
      for r ∈ eachrow(records[f][1:5,:,c])
        @assert all(r .≈ Θ[f])
      end
    end
    @assert all(records[f][6:end,:,:] .≡ missing)
  end


  #test expansion
  maxrecords=(records.numrecords*2.5) |> Int
  samplerecyclefrac=0.5
  sampleexpandfrac=1.0


  #test a single unconstrained expansion
  [record!(dgp; chainnum=c, debugnow=true) for i ∈ 6:records.numrecords, c ∈ 1:records.numchains]
  firstΘ = dgp.Θ |> deepcopy
  dgp.Θ.x .+= 1.0
  originalnumrecords = numrecords
  records = expandchainrecords(;dgp, samplerecyclefrac, sampleexpandfrac, maxrecords)
  dgp = DGP(dgp; records)
  @assert records.numrecords == (
      2*numsamplerecords + numrecords - Int(round(samplerecyclefrac*numsamplerecords)))
  @assert records.numsamplerecords == numsamplerecords * (1+sampleexpandfrac) |> round |> Int
  @assert records.numburnrecords == numburnrecords + numsamplerecords*(1-samplerecyclefrac) |> round |> Int

  [record!(dgp; chainnum=c, debugnow=true) for i ∈ (1+numrecords):records.numrecords, c ∈ 1:records.numchains]
  for f ∈ fieldstocapture
    for c ∈ 1:records.numchains
      @assert all(records[f] .≡ records[f][:,records.chainfieldindex[f],:])
      for r ∈ eachrow(records[f][(numrecords+1):(records.numrecords),:,c])
        @assert all(r .≈ dgp.Θ[f])
      end

      for r ∈ eachrow(records[f][1:numrecords,:,c])
        @assert all(r .≈ firstΘ[f])
      end
    end
  end

  #test a second expansion, constrained
  numrecords = records.numrecords
  numsamplerecords = records.numsamplerecords
  numburnrecords = records.numburnrecords
  secondΘ = dgp.Θ |> deepcopy
  dgp.Θ.x .+= 2.0
  records = expandchainrecords(;dgp, samplerecyclefrac, sampleexpandfrac, maxrecords)
  dgp = DGP(dgp; records)
  
  @assert records.numrecords == maxrecords
  @assert records.numsamplerecords == max(records.numrecords - numrecords + numsamplerecords * samplerecyclefrac,numsamplerecords)  |> round |> Int
  @assert records.numburnrecords == maxrecords - records.numsamplerecords

  [record!(dgp; chainnum=c, debugnow=true) for i ∈ (1+numrecords):records.numrecords, c ∈ 1:records.numchains]
  for f ∈ fieldstocapture
    for c ∈ 1:records.numchains

      #test all three variants
      @assert all(records[f] .≡ records[f][:,records.chainfieldindex[f],:])
      for r ∈ eachrow(records[f][(numrecords+1):(records.numrecords),:,c])
        @assert all(r .≈ dgp.Θ[f])
      end
      
      for r ∈ eachrow(records[f][(originalnumrecords+1):numrecords,:,c])
        @assert all(r .≈ secondΘ[f])
      end

      for r ∈ eachrow(records[f][1:originalnumrecords,:,c])
        @assert all(r .≈ firstΘ[f])
      end
    end
  end





  return nothing

end
