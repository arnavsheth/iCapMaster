
#entry point to acquire the parameters
function getparameters!(parameterfile::String=PARAMETER_FILE)

  #this will contain the parameters
  workbook = XLSX.readxlsx(parameterfile)
  (length(PARAM) > 0) && empty!(PARAM)

  #iterate across sheets
  for sheet::XLSX.Worksheet ∈ workbook.workbook.sheets
    sheetname = "$(sheet.name)"
    (startswith(sheetname, "dep_") || startswith(sheetname, "ref_") )&& continue #allows for depreciaiton of worksheets

    getparameters(sheet, PARAM)
  end

  close(workbook)

  logparameters()

  #PARAM[] = param

  return nothing
end

#extracts and parses the parameters out of each workbook
function getparameters(sheet::XLSX.Worksheet, param::AbstractDict{Symbol, Any})

  #create a dataframe
  paramdf::DataFrame = DataFrame(XLSX.gettable(sheet))

  #println(describe(paramdf))

  #easier to access the vectors as a dictionary
  #paramdfindex = Dict(r.x2=>r.x1 for r ∈ eachrow(paramdf))

  #get the types
  typestrings::Vector{String} = paramdf.type

  #get the default values
  valuestrings::Vector{String} = paramdf.default .|> string

  #get the categories and create a dict for them if there isn't one already
  categories = paramdf.category .|> s->s ≡ missing ? nothing : Symbol(s)
  for category ∈ unique(categories)
    category ≡ nothing && continue
    haskey(PARAM, category) && continue
    PARAM[category] = Dict{Symbol, Any}()
  end

  #get the labels
  parameterlabels = (Symbol).(paramdf.parameter)

  #now add in the overrides
  for (i, s) ∈ enumerate(paramdf.override)
    if !ismissing(s)
      try
        valuestrings[i] = "$s"
      catch err
        error("Failed to load override. parse failed.
          Label: $(parameterlabels[i]) Type: $(typestrings[i]) Val: $s
          error: $err
          trace:\n$(stacktrace(catch_backtrace()))")
      end
    end
  end



  #now parse the parameters to their correct type
  for (parameterlabel, valuestring, typestring, category) ∈ zip(parameterlabels, valuestrings, typestrings, categories)
    try
      
      T::Type = parseparam(Type, typestring)
      p = parseparam(T, valuestring)
      
      if category ≡ nothing
        #each parameter needs a unique key
        haskey(param, parameterlabel) && error(
          "Multiple parameters found with same key.")

        #core parsing logic
        param[parameterlabel] = p
      else
        #each parameter needs a unique key
        haskey(param[category], parameterlabel) && error(
          "Multiple parameters found in category dict $category with same key.")

        #core parsing logic
        param[category][parameterlabel] = p
      end    

    catch err #throw a meaningful error following failure
      throw("Parameter parse failed.
        Label: $(parameterlabel) Type: $(typestring) Val: $valuestring category: $(something(category, "-"))
        error: $err")
    end
  end
end

####formats a dictionary###
formatdict(d;
title="",
header="\nkeys\tvalues\n",
) = title * header * join(string.(keys(d)) .* ":\t" .* string.(values(d)) .* "\n")


#try to write out all the parameter info used
#WARNING: constants need to be actively added to this list, so try to avoid them
function formatparameters()
  b = IOBuffer()
  write(b, "log from time: $(Dates.format(now(),"yyyymmdd_HHMM"))")
  write(b, "\n*****************\n")
  write(b, formatdict(PARAM, title="PARAM dict:"))
  write(b, "\n*****************\n")
  return String(take!(b))
end


function logparameters(;logpath = "$(PARAM[:testpath])/log",
    logname = "cap-$(Dates.format(now(),"yyyymmdd_HHMM"))")
  paramlog = formatparameters()
  open("$logpath/$logname.txt", "w+") do f
    write(f, paramlog)
  end

  return nothing
end



#parsing scenarios
parseparam(::Type{Nothing}, args...) = nothing
parseparam(::Type{T}, s::String) where T<:Number = something(tryparse(T,s), parseparamalways(T,s))
parseparam(::Type{T}, s::String) where T<:Val= eval(Meta.parse("""Val{$(s)}()"""))
parseparam(::Type{String}, s::String) = s
parseparam(::Type{Union{String,Nothing}}, s::String) = s ≡"nothing" ? nothing : s


parseparam(::Type{Date}, i::Int, dateformat::DateFormat) = parseparam(Date, "$i", dateformat)
parseparam(::Type{DateFormat}, s::String) = DateFormat(s)
parseparam(::Type{T}, s::String) where T<:DatePeriod = eval(Meta.parse("""$T($(s))"""))
parseparam(::Type{Date}, s::String, dateformat::DateFormat) = Dates.Date(s, dateformat)

#fallback
parseparam(::Type{T}, s::String) where T = parseparamalways(T,s)
function parseparamalways(::Type{T}, s::String) where T
  x::T = eval(Meta.parse("""($(s))"""))
  return x
end