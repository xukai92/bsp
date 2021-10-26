using Random, UnPack, Setfield
import Distributions: logpdf
import BayesianSymbolic: compute_logjoint, compute_logjoint_tracker
import FileIO: save

Base.@kwdef struct Scene{E, T, G<:Union{Vector{Int}, Missing}}
    entity::E
    traj::T
    # `group[i]` is the group ID of `entity[i]`
    # Entities with the same group ID shares the attributes
    group::G = missing
end

"""
A scenario is defined as a set of scenes with a shared set of entities.
The entities for each scene doesn't have to contain all the entites in the scenario.

For example, in a case where there are 3 entities in a scenario,
the first scene contains entity 1 and 2,
the second scene contains entity 2 and 3 and
the last scene contains entity 1, 2 and 3.
"""
Base.@kwdef struct Scenario{S<:Vector{<:Union{Scene, Missing}}, D<:AbstractFloat}
    scenes::S
    dt::D
    nsteps::Int
end

resultsdir(args...) = projectdir("results", args...)

function save(data::Tuple{<:Scenario, <:Any}, dir::String)
    scenario, attribute = data
    wsave("$dir/data.jld2", @strdict(scenario, attribute))
    @unpack scenes, dt, nsteps = scenario
    open("$dir/info.txt", "w") do io
        println(io, "# Meta")
        println(io, "dt: $dt")
        println(io, "nsteps: $nsteps")
        println(io)
        println(io, "# Scenes")
        foreach(1:length(scenes)) do i
            @unpack entity = scenes[i]
            println(io, "## Scene $i")
            foreach(1:length(entity)) do j
                println(io, "### Entity $j")
                println(io, entity[j])
            end 
            println(io)
        end
    end
    for (i, scene) in enumerate(scenes)
        p = visualize(scene)
        savefig(p, "$dir/traj-$i.png")
        anim = visualize(scene; anim=true)
        gif(anim, "$dir/traj-$i.gif"; show_msg=false)
    end
end

Base.@kwdef struct WeightedSample{T<:AbstractFloat, V}
    logweight::T = 0.0
    value::V
end

make_latents(attributes) = [[WeightedSample(logweight=0.0, value=a)] for a in attributes]

function make_vec_state_pair(traj, dt::AbstractFloat, nahead::Int; truevelocity=false)
    pos = cat(map(state -> state.position, traj)...; dims=3)
    p1, p2, p3 = pos[:,:,1:end-nahead-1], pos[:,:,2:end-nahead], pos[:,:,3:end]
    if !truevelocity
        return State(position=p2, velocity=((p2 - p1) / dt)), State(position=p3)
    else
        vel = cat(map(state -> state.velocity, traj)...; dims=3)
        v2, v3 = vel[:,:,2:end-nahead], vel[:,:,3:end]
        return State(position=p2, velocity=v2), State(position=p3, velocity=v3)
    end
end

function compute_rmse(scene::Scene, getforce, dt::AbstractFloat, nahead::Int=-1, islast::Bool=fasle; resolve_nan=true, root=false)
    @unpack entity, traj = scene
    state_curr, state_target = make_vec_state_pair(traj, dt, nahead)
    local rmse
    for i in 1:nahead
        state_curr = run(Euler(dt), getforce, entity, state_curr)
        # If islast=true, only need to run this in the last iteration
        if !islast || i == nahead
            # Posiiton: [D, N, T]
            diff = state_curr.position - state_target.position[:,:,i:end-nahead+i]
            diffsq = diff .* diff
            # Sum over observation dimension
            diffsqsum = dropdims(
                sum(diffsq; dims=1);
                dims=1 # drop the dim that we sum over
            )
            rmse_curr = root ? sqrt.(diffsqsum) : diffsqsum
            if !islast # accumulate if not islast=false
                rmse = i == 1 ? rmse_curr : rmse + rmse_curr
                # We average over nahead in this case to make sure
                # the rmse doesn't scale up with nahead increasing
                if i == nahead
                    rmse /= nahead
                end
            else       # otherwise just use the last iteration
                rmse = rmse_curr
            end
        end
    end
    if resolve_nan
        # Replace NaN with the maximum non-zero RMSE of the rest
        nanidx = isnan.(rmse)
        safeidx = .~nanidx
        max_safeval = sum(safeidx) > 0 ? maximum(rmse[safeidx]) : Inf
        if iszero(max_safeval)
            rmse[nanidx] .= Inf
        else
            rmse[nanidx] .= max_safeval
        end
    end
    return rmse
end

Base.@kwdef struct Likelihood{N<:AbstractFloat}
    # Number of ahead; -1 is treaded as sequential
    nahead::Int
    # Is the likelihood only based on the last state?
    islast::Bool = false
    # Noise level
    nlevel::N = 1.0
    # Whether or not to average over time and entities
    isnormalized::Bool = false
end

function logpdf(like::Likelihood, scenario::Scenario, getforce)
    @unpack nahead, islast, nlevel, isnormalized = like
    @unpack scenes, dt = scenario
    return -mapreduce(+, scenes) do scene
        # Only compute rooted MSE when to compute nRMSE
        rmse = compute_rmse(scene, getforce, dt, nahead, islast; root=isnormalized)
        isnormalized ? mean(rmse) : (sum(rmse) / nlevel)
    end
end

abstract type NeuralModel end
abstract type NeuralForceModel <: NeuralModel end
abstract type NeuralDynamicsModel <: NeuralModel end

using BayesianSymbolic: AbstractDynamics

function BayesianSymbolic.run(
    ::AbstractDynamics, ndm::NeuralDynamicsModel, entity::Entity, state::State, constraint::T=nothing
) where {T}
    position, velocity = ndm(entity, state)
    return State(position=position, velocity=velocity)
end

compute_logjoint(ScenarioModel, worlds, latents, fm, like) =
    compute_logjoint(ScenarioModel, worlds, latents, make_getforce(fm), like)
function compute_logjoint(ScenarioModel, worlds, latents, getforce::T, like) where T<:Union{Function,NeuralDynamicsModel}
    return mapreduce(+, zip(worlds, latents)) do (scenario, samples)
        expect(samples) do attribute
            fully_obsworld = ScenarioModel(
                scenario, attribute, getforce, like
            )
            compute_logjoint(fully_obsworld)
        end
    end / length(worlds)
end

compute_logjoint_tracker(ScenarioModel, worlds, latents, fm, like) =
    compute_logjoint_tracker(ScenarioModel, worlds, latents, make_getforce(fm), like)
function compute_logjoint_tracker(ScenarioModel, worlds, latents, nm::T, like) where T<:Union{Function,NeuralDynamicsModel}
    return mapreduce(+, zip(worlds, latents)) do (scenario, samples)
        expect(samples) do attribute
            fully_obsworld = ScenarioModel(
                scenario, attribute, nm, like
            )
            compute_logjoint_tracker(fully_obsworld)
        end
    end / length(worlds)
end

compute_normrmse(ScenarioModel, worlds, latents, fm, like) =
    compute_normrmse(ScenarioModel, worlds, latents, make_getforce(fm), like)
function compute_normrmse(ScenarioModel, worlds, latents, getforce::T, like) where T<:Union{Function,NeuralDynamicsModel}
    like = @set(like.nahead = 1) # standardarise the metric
    like = @set(like.isnormalized = true)
    return mapreduce(+, zip(worlds, latents)) do (scenario, samples)
        expect(samples) do attribute
            fully_obsworld = ScenarioModel(
                scenario, attribute, getforce, like; noprior=true
            )
            -compute_logjoint(fully_obsworld)
        end 
    end / length(worlds)
end

# TODO Support adding noise to the simulated trajecotry
function simulate(entity, state0, dt, n_steps, getforce)
    trajectory = Vector{State}(undef, n_steps + 1)
    trajectory[1] = state0
    for i in 1:n_steps
        trajectory[i+1] = run(Leapfrog(dt), getforce, entity, trajectory[i])
    end
    return trajectory
end

"Create a new scenario by an updating function."
function newby(update::Function, scenario::Scenario)
    @unpack scenes, dt, nsteps = scenario
    scenes = map(scenes) do scene
        @unpack entity, traj, group = scene
        entity = deepcopy(entity)
        Scene(entity=update(entity, group), traj=traj, group=group)
    end
    return Scenario(scenes, dt, nsteps)
end
