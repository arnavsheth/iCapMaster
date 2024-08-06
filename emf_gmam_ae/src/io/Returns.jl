using Revise

using DataFrames, CSV, Dates

function summarizefrequency(d; sourcefreq, targetfreq, kwargs...)
  if sourcefreq ≡ something(targetfreq, sourcefreq)
    return d
  elseif sourcefreq ≡ :month && targetfreq ≡ :quarter
    return monthlytoquarterly(; monthly=d)
  else
    throw("no known conversion method from $sourcefreq => $targetfreq . Probably need to code it up.")
  end

  @assert false
end

#summarizes monthly data to quaterly
function monthlytoquarterly(;
  monthly,
  Fvalues=setdiff(propertynames(monthly),[:date]), 
  completequarters=true
  )

  
  monthly.yq = monthly.date .|> dt->year(dt)+quarterofyear(dt)/10

  @eval Main monthly=$(monthly |> deepcopy)
  @eval Main Fvalues=$Fvalues
  quarterly = combine(groupby(monthly,[:yq]),
    :date => maximum => :date,
    Fvalues .=> sum .=> Fvalues,
    nrow=>:Nq,
    )
  
  @assert all(quarterly.Nq .|> q->q ∈ [1,2,3])

  if completequarters
    quarterly = quarterly[quarterly.Nq .== 3, :]
  end
  select!(quarterly, Not([:yq, :Nq]))
  quarterly.date .= quarterly.date .|> lastdayofquarter

  #verify the transformation
  validatedates(quarterly.date, frequency=:quarter, validateexactfrequency=true)

  return quarterly

end
