figuredefaults = (;
    #these colors will be chosen first,
    colors=[colorant"black",  colorant"blue", colorant"dimgrey", colorant"#FF5910",#=colorant"#005C5C",=#],
    #colorgen will stay away from anti-colors
    _anticolors=[colorant"white", colorant"red",colorant"pink",
      colorant"lightblue",colorant"lightyellow",colorant"darkred"],
    colorsgen = (i; )-> 
      i ≤ length(figuredefaults.colors) ? figuredefaults.colors[1:i] : [figuredefaults.colors; 
      distinguishable_colors(i-length(figuredefaults.colors), [figuredefaults.colors;figuredefaults._anticolors],dropseed=true)],

    width=raw"0.95\linewidth",
    #height="$(0.5625*0.95*0.65)\\linewidth", 
    height="$(0.95*9/16)\\linewidth",
    axis_lines="left",
    axis_line_style="{-}",
    tick_label_style = raw"{font=\scriptsize}",
    label_style = raw"{font=\scriptsize}",
    legend_style =raw"{font=\scriptsize}",
    y_tick_label_style = raw"{
      font=\small,
      /pgf/number format/.cd,
      fixed relative,
      precision=2,
      /tikz/.cd,
      }",

      x_tick_label_style = raw"{
        font=\small,
        /pgf/number format/.cd,
        fixed relative,
        precision=2,
        /tikz/.cd,
        }",
  )

#generates a plot and axis object to show the MCMC output results
#for a particular parameter
function sequentialplot(d::AbstractDataFrame;
    title,
    colors,
    distcolors=figuredefaults.colorsgen(:chain ∈ propertynames(d) ? length(unique(d.chain)) : 1; ),
    width,
    height,
    Fθ=setdiff(propertynames(d),[:iteration, :chain])[begin],
    xlabel="iteration",
    ylabel=Fθ |> string |> latexify,
    axis_lines,
    axis_line_style,
    label_style,
    tick_label_style,  

    kwargs...
    )

  N=length(unique(d.iteration)) 
  @assert size(d,2) ≡ 2 + (:chain ∈ propertynames(d)) "DataFrame should only include iteration, focal column, and optionally chain"

  #@eval Main d=$d
  #@eval Main distcolors=$distcolors

  if :chain ∈ propertynames(d)
    tabs=[@pgf(Table(
        {
          x="iteration", 
          y=string(Fθ)
        }, 
        @view(d[d.chain .== c,:])))
          for c ∈ unique(d.chain)]
  else
    tabs = [@pgf(Table({x="iteration", y=string(Fθ)}, d))]
  end
  numchains=length(tabs)
  

  p = @pgf Axis(
    {
      width=width,
      height=height,
      xlabel = xlabel,
      ylabel = ylabel,
      axis_lines=axis_lines,
      axis_line_style=axis_line_style,
      label_style=label_style,
      tick_label_style=tick_label_style,      
      xmin=minimum(d.iteration)-1,
      xmax=N,
      title=title,
    },
  )

  for c ∈ 1:numchains
    push!(p, @pgf( 
        PlotInc( {
          color=distcolors[c],
          no_markers,
        },
        tabs[c],
    )))
  end
    
  
  #Create an array of tables to plot

  
  return p
end

#function bandplot(d; )


#this pre-processes the histogram data
#simplest way to create a denisty plot
function distplothist(vals;
  winsorprop=0.00,
  nbins=20,
  histargs=(;),
  normalizehist=true,
  kwargs...)

  valsadj = winsor(vals, prop=winsorprop) |> collect
  minedge = minimum(valsadj)
  maxedge=maximum(valsadj)
  edges = minedge:((maxedge-minedge)/nbins):maxedge
  
  if normalizehist
    hist=normalize(fit(Histogram, valsadj, edges; histargs...),mode=:pdf)
  else
    hist=fit(Histogram, valsadj, edges; histargs...)
  end

  #put the data into a usable format
  x = (hist.edges[1][1:(end-1)] + hist.edges[1][2:end])/2
  d = (;x, density=hist.weights)

  return d
end

#a more complex method (though the code is simpler due to offloading the hard work
#to the package). Normally wouldn't do that but we are just using the below for
#visualization.
function distplotkd(vals;
  winsorprop=0.00,
  #lowerkdbound=-Inf,
  #upperkdbound=Inf,
  kwargs...)

  #Δbuf = (maximum(vals) - minimum(vals))*0.02
  #boundary=(max(minimum(vals)-Δbuf, lowerkdbound), min(maximum(vals)+Δbuf, upperkdbound))
  valsadj = (winsorprop > 0.0) ? winsor(vals, prop=winsorprop) |> collect : vals

  kd = kde_lscv(valsadj; #=boundary=#)

  return kd
end


#plots the histogram using a histogram object
function distplot(d::AbstractDataFrame;
  #histnames=:chain ∈ propertynames(d) ? unique(d.chain) : [1],
  title=nothing,
  colors,
  distcolors=figuredefaults.colorsgen(:chain ∈ propertynames(d) ? length(unique(d.chain)) : 1; ),
  width,
  height,
  Fθ=setdiff(propertynames(d),[:iteration, :chain])[begin],
  Fdensitystat,
  Fdensitystatcalcmethod,
  ylabel="density", 
  densitymethod=:histogram,
  label,
  axis_lines,
  axis_line_style,
  label_style,
  tick_label_style, 
  x_tick_label_style, 
  xupperbound=Inf,
  xlowerbound=-Inf,
  kwargs...)

  #@eval Main hist=$hist
  N=length(unique(d.iteration)) 

  if densitymethod ≡ :histogram
    hists=[distplothist(d[d.chain .== c,Fθ]; kwargs...) for c ∈ unique(d.chain)]
  elseif densitymethod ≡ :kerneldensity
    hists=[distplotkd(d[d.chain .== c,Fθ] .|> Float64; kwargs...) for c ∈ unique(d.chain)]
  else
    throw("Unorecognized densitymethod $densitymethod")
  end
  numchains=length(hists)

  #use the below to set the boundaries
  allxs = reduce(vcat,[hists[c].x for c ∈ 1:numchains])
  alldensities = reduce(vcat,[hists[c].density for c ∈ 1:numchains])

  xmax = min(max(maximum(allxs),0.1),xupperbound)
  xmin = max(min(0.0,minimum(allxs)),xlowerbound)

  #now draw a vertical line for the point estimate
  ymax = maximum(alldensities)*1.05
  ymin = minimum(alldensities)

  #@info label
  p = @pgf Axis(
    {
      width=width,
      height=height,
      xlabel = label,
      ylabel = ylabel,
      axis_lines=axis_lines,
      axis_line_style=axis_line_style,
      label_style=label_style,
      #y_tick_label_style=isdensity ? density_tick_label_style : tick_label_style,  
      yticklabels={},  
      x_tick_label_style=x_tick_label_style, 
      title=title,
      ymax = ymax,
      xmin=xmin,
      xmax=xmax,
      #title_style={
      #  yshift="8pt"},
    },
  )

  for i ∈ 1:numchains
    p= push!(p, @pgf(PlotInc({
      smooth,
      color=[colors; distcolors][i],
      no_markers, 
      style={thick},     
    },
    Table(x=hists[i].x, y=hists[i].density)))
    )
  end

  if Fdensitystat !== nothing

    if Fdensitystatcalcmethod ≡ :mean
      densitystat = d[!, Fdensitystat] |> mean
      densitystatlabel = "E[$label]=$(round(densitystat,digits=3))"    
    elseif Fdensitystatcalcmethod ≡ :meanroot12
      densitystat = d[!, Fdensitystat] |> mean |> sqrt |> x->x*12^0.5
      densitystatlabel = "E[$label]=$(round(densitystat,digits=3))"
    else
      @assert "unknown Fdensitystatcalcmethod $Fdensitystatcalcmethod"
    end



    w = xmax-xmin
    @pgf push!(p, VLine({color = colorant"red"}, densitystat))
    @pgf push!(p, 
      [raw"\node ",
        {
          #above,
          align="left",
          #rotate=90,
          #anchor = "north west",
          #pin="0:{\\scriptsize $label}"
        },
        "at",
        Coordinate(xmax - w*((length(densitystatlabel)/2+6)/100),ymax-(ymax-ymin)*0.07), 
        #Coordinate(densitystat - w*(length(densitystatlabel)/2/100),(ymax-ymin)*0.07), 
          "{\\footnotesize{\\color{red} $densitystatlabel}};"])

  end

  return p


  #display(push!(a,p))
end

#these plots generally should be computed directly from the DGP
function analysisplots(;dgp, 
  plotparams=PARAM[:iodgpanalysisplotparams],
  figformats=PARAM[:iofigformats],
  rid,
  outputpath,
  figurepath = "$outputpath/fig",)

  isempty(plotparams) && return nothing

  @unpack records=dgp 
  d=records.chain |>DataFrame

  figs = Dict()
  #@eval Main plotparams=$plotparams
  #@eval Main records=$records
  #@info "gothere0"

  for Fθgroup ∈ keys(plotparams)
    #@info "gothere0.5"
    #@eval Main g=$Fθgroup
    for Fθ ∈ [expandchainfields(records, [Fθgroup;]);]

      #@info "gothere1"
      if haskey(plotparams[Fθgroup], :label) && (plotparams[Fθgroup].label !== nothing)
        label=plotparams[Fθgroup].label
      else
        label= (haskey(Fθ2name, Fθ) ? Fθ2name[Fθ] : Fθ) |> paramlabel
      end
      #@info "gothere2"
      dθ = @view(d[:,[:iteration,:chain,Fθ]])
      if plotparams[Fθgroup][:Fdensitystat] !== Fθgroup
        dθ = hcat(dθ, @view(d[:, [plotparams[Fθgroup].Fdensitystat]]))
      end
      p = distplot(dθ; merge(figuredefaults, (;label), plotparams[Fθgroup])...)

      plotname = "$(rid)_$(standardizename(label))_dist"

      figs[plotname] = p
    end

  end

  for (plotname, p) ∈ figs, figformat ∈ figformats 
    try
      #note aregument is the opposite order as with figsave in the plots package
      pgfsave("$figurepath/$plotname.$figformat", p)
    catch err
      @info "Failed to construct $plotname. Error:$err"
      throw(err)
    end
  end


end


PARAM_LABELS = Dict(
  Symbol("β[1]")=>"intercept"
)
#these functions check for labeling overrides
paramlabels(FΘ::AbstractVector) = paramlabel.(FΘ)
paramlabel(Fθ::Symbol) = haskey(PARAM_LABELS,Fθ) ? PARAM_LABELS[Fθ] : string(Fθ)
paramlabel(Fθ) = paramlabel(Fθ |> Symbol)

#=function splitchaindf(din::DataFrame)
  d = deepcopy(din)
  =#

function sequentialconvergenceplot(d::AbstractDataFrame;
  Fθ=setdiff(propertynames(d),[:iteration, :chain])[begin], 
  label=Fθ |> paramlabel,
  convergencegraphwindows=PARAM[:ioconvergencegraphwindows],
  numrecords,
  numburnrecords,
  kwargs...)



  N=length(unique(d.iteration)) 
  @assert all((d.iteration |> unique |> sort!) .== (1:N |> collect)) "
    missing iteration vals- iteration values should be sequential e.g. 1:N"
  @assert size(d,2) ≡ 2 + (:chain ∈ propertynames(d)) "
    DataFrame should only include iteration, focal column, and optionally chain"

  #first scrub the windows
  push!(convergencegraphwindows, numrecords)
  convergencegraphwindows = convergencegraphwindows .|> w->min(numrecords, w)
  unique!(convergencegraphwindows)
  Nwindows = length(convergencegraphwindows)


  convergenceplots = @pgf GroupPlot({
    #group_style = { group_size=group_size},
    group_style= {
      group_size = "1 by $Nwindows",
      vertical_sep=raw"0.15\linewidth"},
  })

  for w ∈ convergencegraphwindows
    dwindow = @view(d[d.iteration .≤ w, :])
    #@eval Main kwargs=$kwargs
    p = sequentialplot(dwindow;Fθ, ylabel=label |> latexify,xlabel=nothing,
      merge(figuredefaults, kwargs |> NamedTuple, (;title=
        "\\medskip\\\\ \\scriptsize{\$\\theta=\$$(Fθ |> latexify) ($w Iterations)}"))...)
    
    if (numburnrecords !== nothing) && (w ≥ numburnrecords)
      ymax = maximum(dwindow[!, Fθ])
      ymin = minimum(dwindow[!, Fθ])
      @pgf push!(p, VLine({color = colorant"red"}, numburnrecords))
      @pgf push!(p, 
        [raw"\node ",
          {
            #above,
            align="left",
            #rotate=90,
            #anchor = "north west",
            #pin="0:{\\scriptsize $label}"
          },
          "at",
          Coordinate(numburnrecords - w*0.07,ymax - (ymax-ymin)*0.02), 
            "{\\tiny{\\color{red} sample start}};"])
    end

    push!(convergenceplots, p)
  end

  return convergenceplots

end

#loops through parameters and create convergence plots for them
function parameterconvergence(;
    convergenceparams = PARAM[:ioconvergenceparams],
    dgp=nothing, 
    records=dgp.records,
    dims=dgp.dims,
    d=records.chain |> DataFrame,
    figformats=PARAM[:iofigformats],
    rid,
    outputpath,
    figurepath = "$outputpath/fig",)


  @assert !records.burnt "parameterconvergence function requires unburnt records"
  FΘ=expandchainfields(records, convergenceparams)

  d = records.chain |> DataFrame
  figs = Dict()

  for Fθ ∈ FΘ
    dθ = @view(d[:,[:iteration,:chain,Fθ]])
    label=paramlabel(Fθ)
    p = sequentialconvergenceplot(dθ; label, numrecords=records.numrecords, numburnrecords=records.numburnrecords)

    plotname = "$(rid)_$(label)_seqconverg"

    figs[plotname] = p

  end

  for (plotname, p) ∈ figs, figformat ∈ figformats 
    try
      #note aregument is the opposite order as with figsave in the plots package
      pgfsave("$figurepath/$plotname.$figformat", p)
    catch err
      @info "Failed to construct $plotname. Error:$err"
      throw(err)
    end
  end


end


#need the below method, annoyingly as autocorrelation can't handle missing
function removechainmissingtype(;records::ChainRecords,)
  @unpack fieldstocapture, chainparts, chainid, burnt=records

  @unpack numrecords, numchains, numsamplerecords, numburnrecords=records

  @assert numrecords == size(records.chain.value,1)
  @assert numchains == size(records.chain.value,3)

  @assert all(records.lastrecord .== numrecords) "last record<numrecords. Method cannot handle this case,
    but could be updated to do so."

    #tehse methods create data structures for the records
    function finalizerecordpart(A; strict=true)
      Anomissing = A .|> Real

      @assert  size(A) ≡ size(Anomissing) || "missing values exist in array to be finalized!"
      return Anomissing
    end

  #remove the missing type
  inits=[f=>finalizerecordpart(chainparts[f]) for f ∈ fieldstocapture]

  return ChainRecords(fieldstocapture; inits, chainid, 
    numrecords, numchains, numsamplerecords, numburnrecords, burnt)
end


function cannedplots(;records, dims, 
    cannedplotsparams=PARAM[:iocannedplotsparams],
    figformats=PARAM[:iofigformats],
    rid,
    outputpath,
    figurepath = "$outputpath/fig",
    )

  nomissingrecords = removechainmissingtype(;records)

  @unpack chain, chainfieldindex, fieldstocapture, burnt=nomissingrecords
  burnt || @warn "Constructing canned diagnostic plots on unburnt data- was this intentional?"

  @eval Main nmr = $nomissingrecords


  for k ∈ keys(cannedplotsparams)
    @unpack label, params = cannedplotsparams[k]
    FΘ=expandchainfields(nomissingrecords, params)

    focalchain = nomissingrecords.chain[:,FΘ,:]
    #@eval Main fc = $focalchain
    plots = [
      "canned_$(rid)_$(label)_seqhist" => plot,
      "canned_$(rid)_$(label)_mixeddensity" => mixeddensity,
      "canned_$(rid)_$(label)_trace" => traceplot,
      "canned_$(rid)_$(label)_mean" => meanplot,
      "canned_$(rid)_$(label)_autocor" => autocorplot,
    ]


    for (plotname, pfunc) ∈ plots,  figformat ∈ figformats 
      try
        p=pfunc(focalchain)
        savefig(p, "$figurepath/$plotname.$figformat")
      catch err
        @info "$plotname failed for key $(k)!"
        throw(err)
      end
    end
      

  end
  #note- we may want to separate this out?


  #=@pgf GroupPlot(
    {
      group_style= {
        group_size="1 by 3"
    },
  },
  (a,p)...,a,p,a,p)=#


end



function testcustomplots(;
    numrecords=2000, 
    numchains=3,
    numsamplerecords=numrecords ÷ 3)

  numburnrecords = numrecords-numsamplerecords

  d = DataFrame(x=rand(Normal(),numrecords*numchains), 
    iteration=reduce(vcat, [collect(1:numrecords) for c ∈ 1:numchains]), 
    chain=reduce(vcat, [fill(c,numrecords) for c ∈ 1:numchains])) 
  #x=rand(Normal(),10_000)
  distplot(d; merge(figuredefaults, (;title="test", label="x",
    Fdensitystat=:x,
    Fdensitystatcalcmethod =:mean,
    densitymethod=:kerneldensity))...) |> display  
  #sequentialplot(d; merge(figuredefaults, (;title="test", ylabel="x",))...) |> display
  #=sequentialconvergenceplot(d;
    dims=(;numburnrecords, numsamplerecords), numrecords) |> display=#

end


