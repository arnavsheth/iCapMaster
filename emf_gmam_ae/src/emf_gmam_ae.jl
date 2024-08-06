module emf_gmam_ae
using Revise
using Distributions, Statistics, LinearAlgebra, Random, DataFrames, StatsBase, BenchmarkTools, UnPack
using Colors, LaTeXStrings, Latexify, StatsPlots, PGFPlotsX, KernelDensity
using SparseArrays, XLSX, SpecialFunctions, Dates, QuadGK, FastGaussQuadrature
using Serialization, ProgressMeter, CodecZstd, CSV, DataStructures, MCMCChains, Optim

import Base: Symbol, show, length, iterate, broadcastable, *, @kwdef, getproperty, propertynames 

#set the backend of StatsPlots- requires LATEX
#pgfplotsx()

#https://github.com/clintonTE/Finometrics#master

Random.seed!(1111)


##########Globals##########################
PARAMETER_FILE = "/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/GMAM/Excel/gmamaeparameters.xlsx"
PARAM::Dict{Symbol, Any} = Dict{Symbol, Any}()

#=BAND_STYLE = raw"\tikzset{
  error band/.style={fill=blue},
  error band style/.style={
      error band/.append style="#1"
  }
}"
if BAND_STYLE ∉ PGFPlotsX.CUSTOM_PREAMBLE
  empty!(PGFPlotsX.CUSTOM_PREAMBLE)
  push!(PGFPlotsX.CUSTOM_PREAMBLE,BAND_STYLE)
end=#
empty!(PGFPlotsX.CUSTOM_PREAMBLE)

########abstract types################
abstract type AbstractModelParameters end
abstract type AbstractModelParametersG <: AbstractModelParameters end
abstract type AbstractDGP{TΘ} end

#I don't think we need to separte abstract model parameters by value type
abstract type AbstractModelParameter end
abstract type AbstractVBParameter <: AbstractModelParameter end

abstract type AbstractDims end
abstract type AbstractData{Tcontainsdata} end

#######################Convenience Types#################

const NDate = Union{Nothing, Date}
const NSymbol = Union{Nothing, Symbol}


#######################Source Files######################

include("io/Parameters.jl")
include("io/SimulateData.jl")

include("dgp/Dims.jl")
include("dgp/DGPModelParameters.jl")
include("dgp/ChainRecords.jl")
include("dgp/DGP.jl")
include("dgp/MeanModel.jl")
include("dgp/Transformations.jl")
include("dgp/Moments.jl")
include("dgp/Priors.jl")
include("dgp/Derived.jl")
include("dgp/CV.jl")
include("dgp/Predict.jl")



include("mcmc/UpdatePosteriors.jl")
include("mcmc/UpdatePosteriorsG.jl")
include("mcmc/UpdatePosteriorsGT.jl")
include("mcmc/UpdatePosteriorsGIR.jl")
include("mcmc/PriorUpdatesG.jl")
include("mcmc/Draw.jl")
include("mcmc/MCMC.jl")
include("mcmc/StopRules.jl")
include("io/ChainStatistics.jl")
include("io/MCMCAnalysis.jl")
include("io/ChainPlots.jl")

include("io/Preprocessing.jl")
include("io/Returns.jl")
include("io/SummaryPlots.jl")
include("vb/ModelParameter.jl")
include("vb/UpdateApproxPosteriors.jl")
include("vb/UpdateApproxPosteriorsG.jl")
include("vb/UpdateApproxPosteriorsGT.jl")
include("vb/UpdateApproxPosteriorsGIR.jl")

include("Utilities.jl")



end # module
