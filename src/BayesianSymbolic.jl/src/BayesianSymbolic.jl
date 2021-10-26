__precompile__(false)   # stop Julia from precompiling our under-dev package
module BayesianSymbolic

export BayesianSymbolic

### Bayesian symbolic infrastructure

using UnPack, LinearAlgebra

include("world.jl")
export Disc, RectMat, RectWall, CircleMat, Spring, Entity, State

## Pretty print for basic structs

Base.show(io::IO, disc::Disc) = 
    print(io, "Disc(radius=$(disc.radius))")
Base.show(io::IO, rm::RectMat) = 
    print(io, "RectMat(width=$(rm.width), height=$(rm.height))")
Base.show(io::IO, rw::RectWall) = 
    print(io, "RectWall(width=$(rw.width), height=$(rw.height))")
Base.show(io::IO, cm::CircleMat) = 
    print(io, "CircleMat(radius_x=$(cm.radius_x), radius_y=$(cm.radius_y))")

function Base.show(io::IO, entity::Entity{Bool})
    @unpack dynamic, mass, magnetism, charge, friction, shape = entity
    print(io, "Entity(\n    dynamic = $dynamic,\n    mass = $mass,\n    magnetism = $magnetism,\n    charge = $charge,\n    friction = $friction,\n    shape = $shape\n)")
end

include("constraints.jl")
export DistanceConstraint, ContactConstraint, SegmantConstraint, solve

include("forces.jl")
export getforce_pairwise

import Base: run
include("dynamics.jl")
export Euler, Aristotle, Leapfrog, run

### Probabilistic modelling

using Random: AbstractRNG
using Distributions: ContinuousMatrixDistribution, MvNormal
using DistributionsAD: arraydist
import Distributions: size, logpdf, rand

struct NoisyPosition{
    PT<:Union{AbstractArray{<:Real, 2}, AbstractArray{<:Real, 3}}, ST
} <: ContinuousMatrixDistribution
    position::PT
    sigma::ST
end

_logpdf_np(position, sigma, x) = 
    logpdf(arraydist([MvNormal(position[:,i], sigma) for i in 1:size(position, 2)]), x)

function logpdf(np::NoisyPosition{<:AbstractArray{<:Real, 2}}, x::AbstractArray{<:Real, 2})
    @unpack position, sigma = np
    return _logpdf_np(position, sigma, x)
end

function logpdf(np::NoisyPosition{<:AbstractArray{<:Real, 3}}, x::AbstractArray{<:Real, 3})
    @unpack position, sigma = np
    position, x = reshape(position, size(position, 1), :), reshape(x, size(x, 1), :)
    return _logpdf_np(position, sigma, x)
end

function rand(rng::AbstractRNG, np::NoisyPosition{<:AbstractArray{<:Real, 2}})
    @unpack position, sigma = np
    return rand(rng, arraydist([MvNormal(position[:,i], sigma) for i in 1:size(position, 2)]))
end

export NoisyPosition

### Turing.jl integration

using Turing: VarInfo, logjoint
using Turing.DynamicPPL: Metadata, AbstractVarInfo, ThreadSafeVarInfo, acclogp!
using ForwardDiff: Dual
using Tracker: TrackedReal

function ThreadSafeVarInfo(vi::AbstractVarInfo)
    return ThreadSafeVarInfo(vi, [Ref{Real}(0e0) for _ in 1:Threads.nthreads()])
end

compute_logjoint(m) = logjoint(m, VarInfo(Metadata(), Ref{Real}(0e0), Ref(0)))

function compute_logjoint_tracker(m)
    vi = VarInfo(Metadata(), Ref(TrackedReal(0e0)), Ref(0))
    return logjoint(m, vi)
end

export compute_logjoint, compute_logjoint_tracker

### ExprOptimization.jl integration

function apply(symtab, ex, nt::NamedTuple)
    foreach(keys(nt)) do k
        symtab[k] = nt[k]
    end
    return Core.eval(symtab, ex)
end

using ExprOptimization: ExprRules, get_executable, ProbabilisticExprRules, RuleNode, return_type
import Distributions: logpdf

const ProbabilisticGrammar = ProbabilisticExprRules.ProbabilisticGrammar

function print_tree(io::IO, tree, grammar)
    ex = get_executable(tree, grammar)
    ExprRules.AbstractTrees.print_tree(io, ex)
end

print_tree(tree, grammar) = print_tree(stdout, tree, grammar)

function logpdf(pcfg::ProbabilisticGrammar, tree::RuleNode)
    @unpack grammar, probs = pcfg
    T = return_type(grammar, tree)
    ind = something(findfirst(isequal(tree.ind), grammar[T]), 0)
    lp = log(probs[T][ind])
    if !isempty(tree.children)
        lp += mapreduce(t -> logpdf(pcfg, t), +, tree.children)
    end
    return lp
end

export apply, print_tree, ProbabilisticGrammar

### Plot.jl integration

using RecipesBase
include("recipes.jl")

### Test Revise.jl

revise_test() = 1
export revise_test

end # module
