# Symbolic regression with ExprOptimization.jl

using ExprOptimization: Grammar, @grammar, get_executable, SymbolTable
using ExprOptimization.ProbabilisticExprRules: RuleNode, mindepth_map, ProbabilisticGrammar
using BayesianSymbolic: contact_point

bnorm(ex) = sqrt.(sum(ex.^2; dims=1))
function bnormalize(ex)
    vnorm = bnorm(ex)
    return ifelse.(iszero.(vnorm), ex, ex ./ vnorm)
end
bdot(ex1, ex2) = sum(ex1 .* ex2; dims=1)
bproject(ex1, ex2) = sum(abs, ex1 .* ex2; dims=1)
bpow2(ex) = ex.^2
bpow3(ex) = ex.^3
bsin(ex) = sin.(ex)
bexp(ex) = exp.(ex)
binv(ex) = ifelse.(ex .> 1e-9, inv.(ex), 0)
badd(ex1, ex2) = ex1 .+ ex2
bsub(ex1, ex2) = ex1 .- ex2
bmul(ex1, ex2) = ex1 .* ex2
bdiv(ex1, ex2) = ex1 .* binv(ex2)
does_collide = BayesianSymbolic.does_collide
is_on = BayesianSymbolic.is_within

G_BSP = @grammar begin
    # Constants
    Const = c1 | c2 | c3
    # Friction coefficient
    Unitless = ui | uj | bsub(ui, uj) | badd(ui, uj)
    # Kg
    Kg = mi | mj | badd(mi, mj) | bsub(mi, mj)
    KgSq = bmul(mi, mj) | bpow2(Kg)
    # Basic vectors
    MeterVec = bsub(pi, pj) | bsub(pi, c) | bsub(pj, c)
    MeterSecVec = vi | vj | bsub(vi, vj)
    # Meter
    Meter = bnorm(MeterVec) | bproject(MeterVec, UnitlessVec) #| lj
    MeterSq = bpow2(Meter)
    # Meter per second
    MeterSec = bnorm(MeterSecVec) | bproject(MeterSecVec, UnitlessVec)
    MeterSecSq = bpow2(MeterSec)
    # Translation-invariant vectors
    TransInvVec = MeterVec | MeterSecVec
    # Unitless vectors
    UnitlessVec = bnormalize(TransInvVec) | bdiv(MeterVec, Meter) | bdiv(MeterSecVec, MeterSec)
    # BasicCoeff
    BasicCoeff = Unitless | Kg | KgSq | Meter | MeterSq | MeterSec | MeterSecSq | bsub(Meter, lj)
    BasicCoeff = bdiv(KgSq, Kg) #| bsub(Meter, Meter) | badd(Meter, Meter) | bsub(MeterSq, MeterSq) | badd(MeterSq, MeterSq)
    # Coefficient
    Coeff = BasicCoeff | bmul(BasicCoeff, BasicCoeff) | bdiv(BasicCoeff, BasicCoeff)
    # Force; defined as magnitude * [condition *] direction
    Bool = does_collide(si, pi, sj, pj) | is_on(si, pi, sj, pj) # condition
    BasicForce = Const * bmul(Coeff, UnitlessVec)               # uncondiitonal force
    BasicForce = Const * bmul(bmul(Coeff, UnitlessVec), Bool)   #   conditional force
    Force = BasicForce #| badd(Force, BasicForce)
end

# G_BSP_[dimensional_analysis][translation_invariants]

G_BSP_00 = @grammar begin
    Const = c1 | c2 | c3
    BasicScalar = ui | uj | mi | mj
    Scalar = BasicScalar | badd(BasicScalar, BasicScalar) | bsub(BasicScalar, BasicScalar) | bmul(BasicScalar, BasicScalar) | bdiv(BasicScalar, BasicScalar) | bpow2(BasicScalar) | bnorm(Vec) | bproject(Vec, Direction) | bsub(Scalar, lj)
    BasicVector = pi | pj | c | vi | vj
    Vec = BasicVector | badd(BasicVector, BasicVector) | bsub(BasicVector, BasicVector)
    Direction = bnormalize(Vec) | bdiv(Vec, Scalar)
    Coeff = Scalar | bmul(Scalar, Scalar) | bdiv(Scalar, Scalar)
    # Force; defined as magnitude * [condition *] direction
    Bool = does_collide(si, pi, sj, pj) | is_on(si, pi, sj, pj) # condition
    BasicForce = Const * bmul(Coeff, Direction)                 # uncondiitonal force
    BasicForce = Const * bmul(bmul(Coeff, Direction), Bool)     #   conditional force
    Force = BasicForce #| badd(Force, BasicForce)
end

G_BSP_01 = @grammar begin
    Const = c1 | c2 | c3
    BasicScalar = ui | uj | mi | mj
    Scalar = BasicScalar | badd(BasicScalar, BasicScalar) | bsub(BasicScalar, BasicScalar) | bmul(BasicScalar, BasicScalar) | bdiv(BasicScalar, BasicScalar) | bpow2(BasicScalar) | bnorm(Vec) | bproject(Vec, Direction) | bsub(Scalar, lj)
    Vec = bsub(pi, pj) | bsub(pi, c) | bsub(pj, c) | vi | vj | bsub(vi, vj)
    Direction = bnormalize(Vec) | bdiv(Vec, Scalar)
    Coeff = Scalar | bmul(Scalar, Scalar) | bdiv(Scalar, Scalar)
    # Force; defined as magnitude * [condition *] direction
    Bool = does_collide(si, pi, sj, pj) | is_on(si, pi, sj, pj) # condition
    BasicForce = Const * bmul(Coeff, Direction)                 # uncondiitonal force
    BasicForce = Const * bmul(bmul(Coeff, Direction), Bool)     #   conditional force
    Force = BasicForce #| badd(Force, BasicForce)
end

G_BSP_10 = @grammar begin
    Const = c1 | c2 | c3
    Unitless = ui | uj | bsub(ui, uj) | badd(ui, uj)
    Kg = mi | mj | badd(mi, mj) | bsub(mi, mj)
    KgSq = bmul(mi, mj) | bpow2(Kg)
    MeterVec = pi | pj | c
    MeterSecVec = vi | vj
    Meter = bnorm(MeterVec) | bproject(MeterVec, UnitlessVec)
    MeterSq = bpow2(Meter)
    MeterSec = bnorm(MeterSecVec) | bproject(MeterSecVec, UnitlessVec)
    MeterSecSq = bpow2(MeterSec)
    Vec = MeterVec | MeterSecVec | badd(MeterVec, MeterVec) | badd(MeterSecVec, MeterSecVec) | bsub(MeterVec, MeterVec) | bsub(MeterSecVec, MeterSecVec)
    UnitlessVec = bnormalize(Vec) | bdiv(MeterVec, Meter) | bdiv(MeterSecVec, MeterSec)
    BasicCoeff = Unitless | Kg | KgSq | Meter | MeterSq | MeterSec | MeterSecSq | bsub(Meter, lj)
    BasicCoeff = bdiv(KgSq, Kg)
    Coeff = BasicCoeff | bmul(BasicCoeff, BasicCoeff) | bdiv(BasicCoeff, BasicCoeff)
    # Force; defined as magnitude * [condition *] direction
    Bool = does_collide(si, pi, sj, pj) | is_on(si, pi, sj, pj) # condition
    BasicForce = Const * bmul(Coeff, UnitlessVec)               # uncondiitonal force
    BasicForce = Const * bmul(bmul(Coeff, UnitlessVec), Bool)   #   conditional force
    Force = BasicForce #| badd(Force, BasicForce)
end

G_BSP_11 = G_BSP

gen_getforce_with_constant(grammar, tree; kwargs...) = 
    gen_getforce_with_constant(grammar, get_executable(tree, grammar); kwargs...)

function gen_getforce_with_constant(grammar, ex::Expr; constant_scale=1, mass_scale=1, external=nothing)
    symtab = SymbolTable(grammar)
    function getforce_pair(etti::Entity, si::State, ettj::Entity, sj::State, constants)
        c = contact_point(etti, si, ettj, sj)
        lj = length(ettj.shape)
        cs = constants * constant_scale
        f_shaped = zeros(size(si)...)
        nt = (
            mi = etti.mass * mass_scale,
            mj = ettj.mass * mass_scale,
            ui = etti.friction,
            uj = ettj.friction,
            si = etti.shape,
            sj = ettj.shape,
            pi = si.position,
            pj = sj.position,
            c = c,
            lj = lj,
            vi = si.velocity,
            vj = sj.velocity,
            c1 = cs[1],
            c2 = cs[2],
            c3 = cs[3],
        )
        return f_shaped .+ apply(symtab, ex, nt)
    end
    if !isnothing(external)
        return (e, s, constants) -> getforce_pairwise(e, s, (ei, si, ej, sj) -> getforce_pair(ei, si, ej, sj, constants)) .+ external(e, s)
    else
        return (e, s, constants) -> getforce_pairwise(e, s, (ei, si, ej, sj) -> getforce_pair(ei, si, ej, sj, constants))
    end
end

using ExprOptimization: ExprOptimization
using ExprOptimization.CrossEntropys: CrossEntropy, Grammar, RuleNode, iseval, mindepth_map,
    initialize, _add_result!, evaluate!, fit_mle!, ExprOptResult, sortperm, _update_tracker!
using ProgressMeter: Progress, next!

function ExprOptimization.CrossEntropys.cross_entropy(
    p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function; verbose::Bool=false
)
    iseval(grammar) && error("Cross-entropy does not support _() functions in the grammar")

    dmap = mindepth_map(grammar)
    losses = Vector{Float64}(undef, p.pop_size)
    pcfg = ProbabilisticGrammar(grammar)
    pop = initialize(p.init_method, p.pop_size, pcfg, typ, dmap, p.max_depth)
    best_tree, best_loss = evaluate!(
        p, loss, grammar, pop, losses, RuleNode(0), Inf; verbose=verbose, pp=(0, p.iterations)
    )
    for iter = 1:p.iterations
        verbose && println("iterations: $iter of $(p.iterations)")
        fit_mle!(pcfg, pop[1:p.top_k], p.p_init)
        for i in eachindex(pop)
            pop[i] = rand(RuleNode, pcfg, typ, dmap, p.max_depth)
        end
        best_tree, best_loss = evaluate!(
            p, loss, grammar, pop, losses, best_tree, best_loss; verbose=verbose, pp=(iter, p.iterations)
        )
    end
    alg_result = Dict{Symbol,Any}()
    _add_result!(alg_result, p.track_method)
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), alg_result)
end

function ExprOptimization.CrossEntropys.evaluate!(
    p::CrossEntropy, loss::Function, grammar::Grammar, pop::Vector{RuleNode},
    losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64;
    verbose::Bool=false, pp=nothing
)

    # losses[:] = loss.(pop, Ref(grammar))

    # Build progress bar description
    pdesc = "Cross-Entropy"
    if !isnothing(pp)
        pdesc *= " ($(join(pp, '/')))"
    end
    pdesc *= ": "
    progress = Progress(length(losses), pdesc)
    #Threads.@threads
    for i in 1:length(losses)
        #losses[i] = loss(pop[i], grammar)
        losses[i] = try
            loss(pop[i], grammar)
        catch e
            if isa(e, InterruptException)
                throw(e)
            else
                Inf
            end
        end
        next!(progress)
    end
    if verbose
        if any(isnan.(losses)) || any(isinf.(losses))
            @info losses
        end
    end

    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
        if verbose
            @info "best_loss=$best_loss"
        end
    end
    _update_tracker!(p.track_method, pop, losses)
    (best_tree, best_loss)
end
