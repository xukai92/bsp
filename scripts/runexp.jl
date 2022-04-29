using InteractiveUtils, DrWatson, Comonicon
if isdefined(Main, :IJulia) && Main.IJulia.inited
    using Revise
else
    ENV["GKSwstype"] = 100 # suppress warnings during gif saving
end
versioninfo()
@quickactivate

using Plots, ProgressMeter, WeightsAndBiasLogger
theme(:bright; size=(300, 300))

include(scriptsdir("bsp.jl"))
@includeall

isdebugbsp() = "JULIA_DEBUG" in keys(ENV) && ENV["JULIA_DEBUG"] == "bsp"

const CE_PRESET      = CrossEntropy(1000, 4, 10, 800, 200.0)
const CE_PRESET_WEAK = CrossEntropy( 800, 3, 10, 600, 200.0)
const CE_DEBUG       = CrossEntropy(  20, 2, 10,  15,   5.0)

function loadpreset(dataset; weak::Bool=false, priordim::Bool=true, priortrans::Bool=true, monly::Bool=false)
    local grammar
    if priordim == true
        if priortrans == true
            grammar = G_BSP_11
        else
            grammar = G_BSP_10
        end
    else
        if priortrans == true
            grammar = G_BSP_01
        else
            grammar = G_BSP_00
        end
    end
    ce = isdebugbsp() ? CE_DEBUG : 
                 dataset == "synth/mat" || weak ? CE_PRESET_WEAK : 
                 CE_PRESET
    preset = Dict(
        "synth/magnet" => (
            ScenarioModel=MagnetScenario, 
            latentname=["fric2", "magn3"],
        ),
        "synth/nbody" => (
            ScenarioModel=NBodyScenario, 
            latentname=["mass1", "mass2", "mass3"],
            ealg = ImportanceSampling(nsamples=200, rsamples=3),
            malg = BSPForce(grammar=grammar, opt=ce, beta=1e-3, ntrials=(monly ? 2 : 1)),
            elike = Likelihood(nahead=5, nlevel=0.01),
            mlike = Likelihood(nahead=1, nlevel=0.01),
        ),
        "synth/bounce" => (
            ScenarioModel=BounceScenario,
            latentname=["mass1", "mass2", "mass3", "mass4"],
            ealg = ImportanceSampling(nsamples=200, rsamples=3),
            malg = BSPForce(grammar=grammar, opt=ce, beta=1e-6, ntrials=(monly ? 4 : 1)),
            elike = Likelihood(nahead=5, nlevel=0.01),
            mlike = Likelihood(nahead=1, nlevel=0.01),
        ),
        # FIXME latentname is wrong
        "synth/mat" => (
            ScenarioModel=MatScenario, 
            latentname=["fric2", "magn3"],
            ealg = ImportanceSampling(nsamples=200, rsamples=3),
            malg = BSPForce(grammar=grammar, opt=ce, beta=1e-3, ntrials=1),
            elike = Likelihood(nahead=5, nlevel=0.001),
            mlike = Likelihood(nahead=1, nlevel=0.001),
        ),
        "phys101/fall" => loadpreset_fall(),
        "phys101/spring" => loadpreset_spring(),
    )
    return preset[dataset]
end

@cast function eonly(
    dataset::String, ntrains::Int;
    shuffleseed::Int=-1,
    slient::Bool=false, 
)
    scenarios, attributes = loaddata(datadir(dataset), ntrains; shuffleseed=shuffleseed, verbose=!slient)
    @unpack ScenarioModel, latentname, ealg, malg, elike, mlike = loadpreset(dataset)
    
    # E-step with zero force
    latents_zero = estep(ealg, ScenarioModel, scenarios, ZeroForce(), elike; verbose=true)
    est_zero = expect.(x -> x, latents_zero)
    mse_zero = mean(map(d -> d[1]^2, est_zero - attributes))
    
    # E-step with oracle force
    latents_oracle = estep(ealg, ScenarioModel, scenarios, OracleForce(), elike; verbose=true)
    est_oracle = expect.(x -> x, latents_oracle)
    mse_oracle = mean(map(d -> d[1]^2, est_oracle - attributes))
    
    !slient && @info "E-step only" mse_zero mse_oracle
    return (zero=mse_zero, oracle=mse_oracle)
end

#@cast function monly(
#    dataset::String, ntrains::Int;
#    seed::Int=0, shuffleseed::Int=-1, nopriordim::Bool=false, nopriortrans::Bool=false,
#    slient::Bool=false, nosave::Bool=false, depthcount::Int=5,
#)
#    hps = @ntuple(ntrains, seed, shuffleseed, nopriordim, nopriortrans)

#    scenarios, attributes = loaddata(datadir(dataset), ntrains; shuffleseed=shuffleseed, verbose=!slient)
#    @unpack ScenarioModel, latentname, ealg, malg, elike, mlike =
#        loadpreset(dataset; priordim=~nopriordim, priortrans=~nopriortrans, monly=true)

#    if !slient
#        nexprs = count_expressions(malg.grammar, depthcount, :Force)
#        @info "Num of expressions upto depth $depthcount" nexprs
#    end

#    # M-step with oracle latent
#    Random.seed!(seed)
#    latents = make_latents(attributes) # orcale latents
#    tused = @elapsed force = mstep(malg, ScenarioModel, scenarios, latents, mlike; verbose=true)

#    expr = BayesianSymbolic.get_executable(force.tree, force.grammar)
#    !slient && @info "M-step only" expr tused
#    !nosave && wsave(
#        resultsdir(dataset, savename(hps; connector="-"), "monly.jld2"),
#        @strdict(ScenarioModel, latentname, ealg, malg, elike, mlike, force),
#    )
#end

function init_ogn(n_inp_units=12, n_emb_units=50, n_hid_units=100, n_out_units=2, actf=relu)
    return (nm = OGNForceModel(
        Chain(
            Dense(1 * n_inp_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_emb_units, actf),
        ),
        Chain(
            Dense(2 * n_emb_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_out_units)
        ),
    ), lr = 2f-3, n_passes_per = 2_000)
end

function init_in(N, n_inp_units=13, n_emb_units=50, n_hid_units=100, n_out_units=2, actf=relu)
    return (nm = INDynamicsModel(
        Chain(
            Dense(1 * n_inp_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_emb_units, actf),
        ),
        Chain(
            Dense(2 * n_emb_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_emb_units)
        ),
        Chain(
            Dense(N * n_emb_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, 2 * N * n_out_units)
        ),
    ), lr = 1f-3, n_passes_per = 400)
end


function init_mlpforce(n_inp_units=12, n_hid_units=100, n_out_units=2, actf=relu)
    return (nm = MLPForceModel(
        Chain(
            Dense(2 * n_inp_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_out_units)
        ),
    ), lr = 1f-3, n_passes_per = 2_000)
end

function init_mlpdynamics(N, n_inp_units=11, n_hid_units=100, n_out_units=2, actf=relu)
    return (nm = MLPDynamicsModel(
        Chain(
            Dense(N * n_inp_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, n_hid_units, actf),
            Dense(1 * n_hid_units, 2 * N * n_out_units)
        ),
    ), lr = 1f-3, n_passes_per = 400)
end

function loadpreset_neural(dataset, model)
    n_entities = Dict("synth/nbody"=>1+3, "synth/bounce"=>4+4, "synth/mat"=>2)[dataset]
    preset_model = Dict(
        "ogn" => init_ogn(),
        "in" => init_in(n_entities),
        "mlpforce" => init_mlpforce(),
        "mlpdynamics" => init_mlpdynamics(n_entities),
    )
    preset = Dict(
        "synth/nbody" => (
            ScenarioModel=NBodyScenario,
            latentname=["mass1", "mass2", "mass3"],
            mlike = Likelihood(nahead=1, nlevel=1e-2, isnormalized=true),
            preset_model[model]...
        ),
        "synth/bounce" => (
            ScenarioModel=BounceScenario,
            latentname=["mass1", "mass2", "mass3", "mass4"],
            mlike = Likelihood(nahead=1, nlevel=1e-2, isnormalized=true),
            preset_model[model]...
        ),
        "synth/mat" => (
            ScenarioModel=MatScenario,
            latentname=["fric2", "magn3"],
            mlike = Likelihood(nahead=1, nlevel=1e-3, isnormalized=true),
            preset_model[model]...
        ),
    )
    return preset[dataset]
end

@cast function monly_neural(
    dataset::String, ntrains::Int, model::String;
    seed::Int=0, shuffleseed::Int=-1,
    slient::Bool=false, logging::Bool=false, nosave::Bool=false,
)
    hps = @ntuple(ntrains, seed, shuffleseed)
    scenarios, attributes = loaddata(datadir(dataset), ntrains; shuffleseed=shuffleseed, verbose=!slient)
    @unpack ScenarioModel, latentname, nm, mlike, lr, n_passes_per = loadpreset_neural(dataset, model)

    # M-step with oracle latent
    Random.seed!(seed)
    latents = make_latents(attributes) # orcale latents

    n_passes = n_passes_per * length(scenarios)
    isdebugbsp() && (n_passes = div(n_passes, 100)) # fast run when debugging
    # TODO Implement logging in mstep!
    tused = @elapsed nm, ptrace = mstep!(nm, ScenarioModel, scenarios, latents, mlike, n_passes, lr; verbose=true)

    res = @strdict(ScenarioModel, latentname, nm, mlike, ptrace)

    !slient && @info "M-step only (neural)" tused
    !nosave && wsave(
        resultsdir(dataset, savename(hps; connector="-"), "monly-$model.jld2"), res
    )

    return res
end

@cast function em(
    dataset::String, ntrains::Int, niters::Int; 
    seed::Int=0, shuffleseed::Int=-1, idx::Int=0,
    slient::Bool=false, logging::Bool=false, nosave::Bool=false,
)
    hps = @ntuple(ntrains, seed, shuffleseed)

    idcs = idx == 0 ? nothing : [idx]
    scenarios, attributes = loaddata(datadir(dataset), ntrains; idcs=idcs, shuffleseed=shuffleseed, verbose=!slient)
    @unpack ScenarioModel, latentname, ealg, malg, elike, mlike = loadpreset(dataset; weak=true)
    
    logging && (logger = WBLogger(project="BSP"))
    logging && config!(logger, hps)
    
    # EM
    Random.seed!(seed)
    trace = []
    force = ZeroForce()
    slient && (progress = Progress(niters, "Exp-Max"))
    for iter in 1:niters
        !slient && @info "Exp-Max ($iter/$niters)"
        latents = estep(ealg, ScenarioModel, scenarios, force, elike; verbose=!slient)
        force = mstep(malg, ScenarioModel, scenarios, latents, mlike; verbose=!slient)        
        logging && with_logger(logger) do
            nrmse = compute_normrmse(ScenarioModel, scenarios, latents, force, mlike)
            @info "train" iter=iter normrmse=nrmse
        end
        slient && next!(progress)
        push!(trace, (latents=latents, force=force))
    end
    
    expr = BayesianSymbolic.get_executable(force.tree, force.grammar)
    !slient && @info "EM" expr force.constant
    !nosave && wsave(
        resultsdir(dataset, savename(hps; connector="-"), "em.jld2"), 
        @strdict(ScenarioModel, latentname, ealg, malg, elike, mlike, trace),
    )
end

@main
