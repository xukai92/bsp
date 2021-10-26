using InteractiveUtils, DrWatson
versioninfo()
@quickactivate

# Ref: https://github.com/julia-vscode/julia-vscode/issues/1076#issuecomment-624732827
ENV["GKSwstype"] = "100" # allow GR to be used in terminal

using Comonicon, ProgressMeter, Parameters
using Random, LinearAlgebra, Turing, BayesianSymbolic
using Plots
theme(:bright; size=(300, 300))

Random.seed!(1110) # fix the random seed for reproducibility

# Global simulation configurations
const SIM = (
        dt = 2e-2,  # step size of integrator
    nsteps = 50,    # #{discretization steps}
    nlevel = 1e-2,  # observation noise level
)

include(srcdir("utility.jl"))
include(srcdir("analyse.jl"))

# Magnet

include(srcdir("scenarios/magnet.jl"))

@cast function magnet(nscenarios::Int, nscenes::Int=1)
    genscenario = MagnetScenario(
        Scenario(scenes=fill(missing, nscenes), dt=SIM.dt, nsteps=SIM.nsteps), 
        missing, 
        BayesianSymbolic.getforce,
        Likelihood(nahead=1, nlevel=SIM.nlevel),
    )

    progress = Progress(nscenarios)
    Threads.@threads for i in 1:nscenarios
        scenario, attribute = genscenario()
        save((scenario, attribute), datadir("synth", "magnet", lpad(i, 3, "0")))
        next!(progress)
    end
end

# N-body

include(srcdir("scenarios/nbody.jl"))

@cast function nbody(nscenarios::Int, nscenes::Int=1; nbodies::Int=4)
    getforce = (e, s) -> BayesianSymbolic.getforce_pairwise(e, s; G=G_NBODY)
    genscenario = NBodyScenario(
        Scenario(scenes=fill(missing, nscenes), dt=SIM.dt, nsteps=SIM.nsteps),
        missing,
        getforce,
        Likelihood(nahead=1, nlevel=SIM.nlevel);
        nbodies=nbodies,
    )

    progress = Progress(nscenarios)
    Threads.@threads for i in 1:nscenarios
        scenario, attribute = genscenario()
        save((scenario, attribute), datadir("synth", "nbody", lpad(i, 3, "0")))
        next!(progress)
    end
end

# Bounce

include(srcdir("scenarios/bounce.jl"))

@cast function bounce(nscenarios::Int, nscenes::Int=1; ndiscs::Int=4)
    genscenario = BounceScenario(
        Scenario(scenes=fill(missing, nscenes), dt=SIM.dt, nsteps=SIM.nsteps),
        missing,
        BayesianSymbolic.getforce,
        Likelihood(nahead=1, nlevel=SIM.nlevel);
        ndiscs=ndiscs,
    )

    progress = Progress(nscenarios)
    Threads.@threads for i in 1:nscenarios
        scenario, attribute = genscenario()
        save((scenario, attribute), datadir("synth", "bounce", lpad(i, 3, "0")))
        next!(progress)
    end
end

# Mat

include(srcdir("scenarios/mat.jl"))

@cast function mat(nscenarios::Int, nscenes::Int=1; nmats::Int=1)
    genscenario = MatScenario(
        Scenario(scenes=fill(missing, nscenes), dt=SIM.dt, nsteps=SIM.nsteps),
        missing,
        BayesianSymbolic.getforce,
        Likelihood(nahead=1, nlevel=SIM.nlevel);
        nmats=nmats,
    )

    progress = Progress(nscenarios)
    Threads.@threads for i in 1:nscenarios
        scenario, attribute = genscenario()
        save((scenario, attribute), datadir("synth", "mat", lpad(i, 3, "0")))
        next!(progress)
    end
end

#

@main
