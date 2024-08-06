using Revise

using BenchmarkTools, DataFrames, CSV, Pkg, UnPack, Random

if  pwd() ∉ LOAD_PATH
  push!(LOAD_PATH,pwd())
end
#throw("proceed with the convergence diagnostic graphs")
Pkg.activate(pwd())
import emf_gmam_ae as m

#Random.seed!(1111)
function runscripts()
  m.getparameters!() #WARNING- mutates parameter dictionary to match excel parameter file
  
  #dgp = m.loadmcmctest(;) #initializes an mcmc but does not run it  dgp = m.chainanalysis(; dgp)
  #@eval Main dgp=$dgp
  

  #flow for simulate
  #=@unpack dgp, rid = m.simulatechainanalysis(;)
  m.PARAM[:iorid] = rid #WARNING- this breaks "the PARAM only changes w/ the file" integrity, but its a helpful shortcut
  @eval Main dgp=$dgp
  #throw("stop")
  m.analyzemcmcoutput()=#
  #@eval Main dgp=$(dgp)

 # m.testcustomplots()

 
  #m.writedataascsv(;)``
  
 #processing data, changing format, converts to log return (if necessary)
  m.processfactors()
 #making sure input data is the expected format 
  m.processassets()

  #m.Random.seed!(11)
  
  m.batchchainanalysis(; setseed=111)
  #m.testcv()
  return nothing
  
end 


@time runscripts()