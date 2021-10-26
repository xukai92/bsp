@model function SpringScenario(scenario, attribute, getforce, like; mass=missing, noprior=false)
    if noprior
        @assert !ismissing(attribute)
    else
        attribute ~ arraydist([Uniform(2e-2, 2.0)])
    end
    tensor, = attribute
    if eltype(scenario.scenes) <: Missing
        if ismissing(mass)
            mass = rand(Uniform(2e-2, 2e-1))
            mass *= 1e3
        end
        len = rand(truncated(Normal(5.1, 2), 2, 8))
        disc = Entity(dynamic=true, mass=(mass / 1e3), shape=Disc(2e-1))
        spring = Entity(dynamic=false, mass=1e1, shape=Spring(tensor, len, disc))
        entity = cat(disc, spring)

        scenes = map(1:length(scenario.scenes)) do idx
            position ~ arraydist([Normal(0, 0.05), Normal(5, 0.1)])
            velocity ~ arraydist([Normal(0, 0.05), Normal(0, 0.1)])
            state0 = State(position=hcat(-position, zeros(2)), velocity=hcat(velocity, zeros(2)))
            traj = simulate(entity, state0, scenario.dt, scenario.nsteps, getforce)
            Scene(entity=entity, traj=traj)
        end
        return Scenario(scenes=scenes, dt=scenario.dt, nsteps=scenario.nsteps), attribute
    else
        scenario = newby(scenario) do entity, group
            shape = entity.shape
            entity.shape[2] = Spring(tensor, shape[2].length, shape[2].target)
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end

@model function SpringScenarioUnkownLength(scenario, attribute, getforce, like)
    prior_tensor = Uniform(2e-2, 2.0)
    prior_len = truncated(Normal(5.1, 2), 2, 8)
    prior_attribute = arraydist([prior_tensor, prior_len])
    attribute ~ prior_attribute
    tensor, len = attribute
    if eltype(scenario.scenes) <: Missing
        @assert "No generation supoorted"
    else
        scenario = newby(scenario) do entity, group
            disc = entity[1]
            entity.shape[2] = Spring(tensor, len, disc)
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end

function getforce_pair_spring(ei::Entity, si::State, ej::Entity, sj::State)
    f = zeros(si.velocity |> size)
    if ej.shape isa Spring && ej.shape.target == ei
        Δp = sj.position - si.position
        rsq = sum(abs2, Δp; dims=1)
        u = Δp ./ sqrt.(rsq)
        Δr = ej.shape.length .- sqrt.(rsq)
        f += -ej.shape.tensor * Δr .* u
    end
end

function getforce_spring(entity, state; kwargs...)
    f = getforce_pairwise(entity, state, getforce_pair_spring; kwargs...)
    return f .+ [0, -9.8] .* entity.mass'
end

getforce_spring_external(e, s) = [0, -9.8] .* e.mass'

function loadpreset_spring()
    return (
        ScenarioModel=SpringScenario,
        latentname=["tensor"],
        ealg = ImportanceSampling(nsamples=5_000, rsamples=3),
        malg = BSPForce(
            grammar=G_BSP, opt=CrossEntropy(400, 2, 10, 300, 100.0), beta=1e0, external=getforce_spring_external
        ),
        elike = Likelihood(nahead=10, nlevel=0.1),
        mlike = Likelihood(nahead=3,  nlevel=0.1),
    )
end
