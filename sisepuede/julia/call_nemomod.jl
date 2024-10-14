# activate the environment
using Pkg
Pkg.activate(".")

# load pacakges
using DataFrames
using NemoMod
using SQLite
using Cbc
using JuMP

# include system setup
include("setup_analysis.jl")
