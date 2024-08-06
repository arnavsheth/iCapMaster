#loads the summary files
function loadsummariesforplots(;
    rid=PARAM[:batchiorid],
    analysispath=PARAM[:analysispath],
    fundnames=:all,
    fundnameindex=PARAM[:fundnameindex],
    summarydateformat = dateformat"yyyy-mm-dd"
    )
  
  
    
  
    summaryfiles = readdir("$analysispath/$rid/summary/whole")
    if fundnames ≡ :all
      filter!(f->match(Regex("[a-zA-Z0-9_]_[a-zA-Z0-9_]*$rid.csv"),f) !== nothing, summaryfiles)
      fundlabels = map(summaryfiles) do summaryfile
        replace(summaryfile, "s_A_"=>"", "_$rid.csv"=>"", ) |> Symbol
      end
    else
      fundlabels = fundnames .|> fundname->fundnameindex[fundname] .|> Symbol
      summaryfiles = map(fundlabels) do fundlabel
        singlesummaryfile = filter(f->match(Regex("[a-zA-Z0-9_]*$(fundlabel)_[a-zA-Z0-9_]*$rid.csv"),f) !== nothing
        , summaryfiles)
        if length(singlesummaryfile)>1
          throw("Multiple summary files found for fund for fundname $fundname
            foundfiles: $singlesummaryfile")
        elseif length(summaryfiles) < 1
          @warn "$fundname not found"
        end
  
        return singlesummaryfile[begin]
      end
    end
  
    summarydfs = map(summaryfiles) do summaryfile
      d = CSV.File("$analysispath/$rid/summary/whole/$(summaryfile)") |> DataFrame
      for f ∈ [:chain, :param, :paramgroup,]
        d[!, f] .= d[!, f] .|> Symbol
      end
      chainname = ifelse(length(d.chain |> unique) == 1 , d.chain[begin], :full)
      d = d[d.chain .≡ chainname,:]
      d.date = d.note .|> dt->something(tryparse(Date, dt, summarydateformat), missing)

      sort!(d, [:paramgroup, :date, :param])
      return d

    end
  
    return OrderedDict(fundlabels .|> Symbol .=> summarydfs)
  
  end

  #generates a plot and axis object to show the MCMC output results
#for a particular parameter

#helper function to loop through a data frame for all ofcal parameters
#=function bandplots(d::AbstractDataFrame; θs = PARAM[:focalparams], title)

  ps = map(keys(θs)) do Fθ 
    @unpack Fdata, label = θs[Fθ]
    dθ = d[d.paramgroup .≡ Fθ,:]
    bandplot(dθ; merge(figuredefaults, (;ylabel=label, Fdata, title="$title: $label"))...)
  end

  return ps

end=#

function bandplot(d::AbstractDataFrame;
    title,
    colors,
    distcolors=figuredefaults.colorsgen(10),
    width,
    height,
    Fdata=nothing,
    Fpredicted=PARAM[:bandplot][:predicted],
    Fbands,
    xlabel=nothing,
    ylabel=Fθ |> string |> latexify,
    axis_lines,
    axis_line_style,
    label_style,
    tick_label_style,
    y_tick_label_style,  
    ymax_override=nothing,
    ymin_override=nothing,

    kwargs...
    )


  #@eval Main d=$d
  #@assert (d.date .!== missing) |> all
  @assert issorted(d, :date)

  #@info "title=$title"
  #@info "ymax_override=$ymax_override"
  

  BAND_STYLE = raw"\tikzset{
    error band/.style={fill=blue},
    error band style/.style={
        error band/.append style=#1
    }
  }"
  if BAND_STYLE ∉ PGFPlotsX.CUSTOM_PREAMBLE
    #empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE,BAND_STYLE)
  end

  #empty!(PGFPlotsX.CUSTOM_PREAMBLE)


  cycle_list = [
    @pgf({error_band_style = raw"blue!20"}),
    @pgf({error_band_style = raw"blue!40"}),
    @pgf({error_band_style = raw"blue!60"}),
    @pgf({error_band_style = raw"blue!80"}),
    @pgf({error_band_style = raw"blue!100"}),
  ]

  #@eval Main cycle_list=$cycle_list

  #simple heuristic for computing the axes range
  #@eval Main d=$d
  datavalues = Fdata !== nothing ? d[!,Fdata] |> skipmissing |> collect : Float64[]
  predictedvalues = Fpredicted !== nothing ? d[!,Fpredicted] |> skipmissing |> collect : Float64[]

  #heuristics for axis bounds
  if !isempty(Fbands)
    bandvals = reduce(vcat, d[!,reduce(vcat, Fbands .|> p->[p.first; p.second])] |> eachcol |> collect)
  else
    bandvals = Float64[]
  end

  @assert !isempty([datavalues; predictedvalues; bandvals]) "no y values found- 
    Fpredicted=$(something(Fpredicted,"")), Fbands=$(something(Fbands,"")), Fdata=$(something(Fdata,""))"
  #compute the tick distance
  ymaxraw = something(ymax_override, maximum([datavalues; predictedvalues; bandvals]))
  yminraw = something(ymin_override, minimum([datavalues; predictedvalues; bandvals]))
  Δyboundsraw = ymaxraw-yminraw
 
  if Δyboundsraw < 0.05
    ytick_distance = 0.005
  elseif Δyboundsraw < 0.1
    ytick_distance = 0.01
  elseif Δyboundsraw < 0.2
    ytick_distance = 0.02  
  elseif Δyboundsraw < 0.5
    ytick_distance = 0.05
  elseif Δyboundsraw < 1.0
    ytick_distance = 0.1
  elseif Δyboundsraw < 2.0
    ytick_distance = 0.2  
  elseif Δyboundsraw < 5.0
    ytick_distance = 0.5
  else
    ytick_distance = 1.0
  end

  #allow for axis range overrides
  if ymax_override !== nothing
    ymax = max(ytick_distance*2,ceil(maximum(ymaxraw+ytick_distance)*1/ytick_distance * 100)*ytick_distance/100)
  else
    ymax=ymax_override
  end
  if ymin_override !== nothing
    ymin = min(-ytick_distance*2,floor(minimum(yminraw -ytick_distance)*1/ytick_distance * 100)*ytick_distance/100)
  else
    ymin=ymin_override
  end

  xmin, xmax = minimum(d.date), maximum(d.date)
  if 10*365 < (xmax - xmin).value
    @pgf(xticklabel={"\\year"})
    xtick_distance=365*3   
  elseif  10*365 > (xmax - xmin).value > 3*365
    @pgf(xticklabel={"\\year"})
    xtick_distance=365
  else 3*365 > (xmax - xmin).value
    @pgf(xticklabel={"\\month-\\year"})
    xtick_distance=181
  end      
  

  
  #generally following the technique shown https://pgfplots.net/error-intervals/
  p = @pgf Axis(
    {
      width=width,
      height=height,
      xlabel = xlabel,
      ylabel = ylabel,
      axis_lines=axis_lines,
      axis_line_style=axis_line_style,
      label_style=label_style,
      #tick_label_style=tick_label_style, 
      y_tick_label_style=y_tick_label_style,  
      date_coordinates_in = "x",
      ytick_distance=ytick_distance,
      xticklabel=xticklabel,
      xtick_distance=xtick_distance,
      xmin=xmin,
      xmax=xmax,
      ymin=ymin,#-0.2,
      ymax=ymax,#0.2,
      title=title,
      enlarge_x_limits = false,
      scaled_y_ticks=false,
      cycle_list=cycle_list,
      #=cycle_list = {
        {color="blue!20", fill="blue"},
        {color="blue!40", fill="blue"},
        {color="blue!60", fill="blue"},
        {color="blue!80", fill="blue"},
        {color="blue!100", fill="blue"},
      }=#
    },
  )

  for (Flower,Fupper) ∈ Fbands
    d.dif = d[!,Fupper] - d[!, Flower]
    d.neg2dataPlower = -d[!,Flower]-d.dif

    push!(p, @pgf(
      Plot({
        draw="none",
        stack_plots="y",
        forget_plot,
      },
      Table({x="date", y=string(Flower)}, d[:, [:date, Flower]]))
    ))
    push!(p, @pgf(
      PlotInc({
        draw="none",
        stack_plots="y",
        error_band,
      }, 
      Table({x="date", y="dif"}, d[:, [:date, :dif]]),
      [raw"\closedcycle"],
      )
    ))
    
    push!(p, @pgf(
      Plot({
        draw="none",
        stack_plots="y",
        forget_plot,
      },
      Table({x="date", y="neg2dataPlower"}, d[:, [:date, :neg2dataPlower ]]))
    ))

    select!(d, Not([:dif, :neg2dataPlower]))

    

  end

  if Fdata !== nothing
    push!(p, @pgf(
      Plot({
        #draw="none",
        #"only_marks",
        #mark="triangle*",
        mark="none",
        densely_dashed,
        "darkgray",
        #=linecolor = "darkgray",
        linestyle = "dashed",  =#
        forget_plot,
      },
      Table({x="date", y=string(Fdata)}, d[d[!,Fdata] .!== missing, [:date, Fdata ]]))
    ))
  end

  if Fpredicted !== nothing
    push!(p, @pgf(
      Plot({
        #draw="none",
        #"only_marks",
        #mark="triangle*",
        mark="none",
        #densely_dashed,
        "black",
        #=linecolor = "darkgray",
        linestyle = "dashed",  =#
        forget_plot,
      },
      Table({x="date", y=string(Fpredicted)}, d[:, [:date, Fpredicted ]]))
    ))
  end
    
  
  return p
end


#loop through all funds for the plots
function createbandplots(;
  figformats=PARAM[:iofigformats],
  rid=PARAM[:batchiorid],
  analysispath=PARAM[:analysispath],
  outputpath = "$analysispath/$rid",
  figurepath = "$outputpath/fig",
  θs = PARAM[:bandplot][:focalparams], 
  overrides = PARAM[:bandplot][:focalparamoverrides], 
  title=nothing)
  summaries = loadsummariesforplots(; rid)



  figs = Dict()
  for (k,d) ∈ zip(keys(summaries), values(summaries)), Fθ ∈ keys(θs)
    #look for overrides
    overrideparams=nothing
    Ffocal= θs[Fθ][:Ffocal]

    if Ffocal ∉ d.paramgroup
      @warn ("$Ffocal listed as bandplot key in focalparams but not present in data/derived data. Skipping band plot.")
    end

    for (fundname,fundoverrides) ∈ overrides
      #@info "k=$k, rid=$rid, occursin(k, rid)=$(occursin("$k", rid))"
      if occursin("$fundname", "$k")
        overrideparams = fundoverrides
        continue
      end
    end

    #b = bandplot(v; title=nothing)
    @unpack label = θs[Fθ]
    Fdata = :Fdata ∈ keys(θs[Fθ]) ? θs[Fθ][:Fdata] : nothing
    Fpredicted = :Fpredicted ∈ keys(θs[Fθ]) ? θs[Fθ][:Fpredicted] : nothing
    Fbands = something(:Fbands ∈ keys(θs[Fθ]) ? θs[Fθ][:Fbands] : nothing, PARAM[:bandplot][:bands])
    dθ = d[d.paramgroup .≡ Ffocal,:]

    isempty(dθ) && continue
    #@info "plotting Fθ=$Fθ"
    b=bandplot(dθ; merge(
        figuredefaults, 
        (;ylabel=string(label), Fdata, Fpredicted, Fbands, title), 
        something(overrideparams, (;)))...)

    plotname = "$(rid)_$(k)$(Fθ≡Ffocal ? Fθ : "$(Ffocal)_$(Fθ)")_band"

    figs[plotname] = b
  end


  for (plotname, p) ∈ figs, figformat ∈ figformats 
    try
      #note aregument is the opposite order as with figsave in the plots package
      pgfsave("$figurepath/$plotname.$figformat", p)
    catch err
      @info "Failed to construct $plotname. Error:$err"
      @eval Main p=$p
      throw(err)
    end
  end

  return nothing
end
  

#some utilities to parse the dates without erroring out on a failure
Base.tryparse(::Type{<:Date}, ::Missing, df::DateFormat) = missing
function Base.tryparse(::Type{<:Date}, str::AbstractString, df::DateFormat)
  try
    d = Date(str, df)
    return d
  catch
    return nothing
  end
end
