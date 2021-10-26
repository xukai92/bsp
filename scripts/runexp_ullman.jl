using InteractiveUtils, DrWatson, Comonicon
if isdefined(Main, :IJulia) && Main.IJulia.inited
    using Revise
else
    ENV["GKSwstype"] = 100 # suppress warnings during gif saving
end
versioninfo()
@quickactivate

using Plots, ProgressMeter, Logging, WeightsAndBiasLogger
theme(:bright; size=(300, 300))

using Random, Turing, BayesianSymbolic
using ExprOptimization.ExprRules

includef(args...) = isdefined(Main, :Revise) ? includet(args...) : include(args...)
includef(srcdir("utility.jl"))
includef(srcdir("app_inf.jl"))
includef(srcdir("sym_reg.jl"))
includef(srcdir("network.jl"))
includef(srcdir("exp_max.jl"))
includef(srcdir("analyse.jl"))
includef(srcdir("dataset.jl"))
# Suppress warnings of using _varinfo
with_logger(SimpleLogger(stderr, Logging.Error)) do
    includef(srcdir("scenarios", "ullman.jl"))
end

includef(scriptsdir("ullman_hacks.jl"))

@main function main(
    wid::Int, sid::Int, niters::Int;
    seed::Int=0,
    slient::Bool=false, logging::Bool=false, nosave::Bool=false,
)
    ScenarioModel = UllmanScenario

    hps = @ntuple(wid, sid, niters, seed)

    # Load World 1 for fitting external
    # NOTE World 1 has only collision and stuff
    scenario_ext, attribute_ext = loadullman(datadir("ullman", "processed"), 1)

    malg_ext = HandCodedForce(niterations=3, mask=Bool[1,1,0,0,1])
    mlike_ext = Likelihood(nahead=3, nlevel=0.1)
    latents_ext = make_latents([attribute_ext])
    force_ext = mstep(malg_ext, ScenarioModel, [scenario_ext], latents_ext, mlike_ext; verbose=!slient)

    # Load the training data
    scenario, _ = loadullman(
        datadir("ullman", "processed"), wid; idcs=[sid]
    )
    scenarios = [scenario]

    latentname = ["mass1", "mass2", "mass3", "charge1", "charge2", "charge3", "fric1", "fric2", "fric"]
    ealg = ImportanceSampling(nsamples=200, rsamples=3)
    elike = Likelihood(nahead=5, nlevel=0.1)
    mlike = mlike_ext

    malg = BSPForce(
        grammar=G_BSP_ULLMAN, opt=CrossEntropy(400, 2, 10, 300, 100.0), beta=1e-1, external=make_getforce(force_ext)
    )

    # EM
    Random.seed!(seed)
    trace = []
    force = ZeroForce()
    slient && (progress = Progress(niters, "Exp-Max"))
    for iter in 1:niters
        !slient && @info "Exp-Max ($iter/$niters)"
        latents = estep(ealg, ScenarioModel, scenarios, force, elike; verbose=!slient)
        force = mstep(malg, ScenarioModel, scenarios, latents, mlike; verbose=!slient)
        slient && next!(progress)
        push!(trace, (latents=latents, force=force))
    end

    expr = BayesianSymbolic.get_executable(force.tree, force.grammar)
    !slient && @info "EM" expr force.constant
    !nosave && wsave(
        resultsdir("ullman", savename(hps; connector="-"), "em.jld2"),
        @strdict(ScenarioModel, latentname, ealg, malg, elike, mlike, trace)
    )
end
