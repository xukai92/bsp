@model function MatScenario(
    scenario, attribute, getforce, like; noprior=false, nmats=missing
)
    if ismissing(nmats)
        nmats = length(first(scenario.scenes).entity) - 1
    end
    if noprior
        @assert !ismissing(attribute)
    else
        #prior_attribute = filldist(Uniform(5, 20), nmats)
        friction_true = scenario.scenes[1].entity.friction[2:end]
        prior_attribute = arraydist(
            [DiscreteNonParametric([f/3, f/2, f, 2f, 3f], ones(5) / 5) for f in friction_true]
        )
        attribute ~ prior_attribute
    end
    friction = attribute
    if eltype(scenario.scenes) <: Missing
        dynamic = [true, fill(false, nmats)...]
        mass ~ Uniform(2e-2, 9)
        mass = [mass, ones(nmats)...]
        friction = [0, friction...]
        width ~ filldist(Uniform(20, 50), nmats)
        height ~ filldist(Uniform(20, 50), nmats)
        shape = [Disc(5e0), RectMat.(width, height)...]
        entity = Entity(dynamic=dynamic, friction=friction, mass=mass, shape=shape)

        scenes = map(1:length(scenario.scenes)) do idx
            angle ~ filldist(Uniform(0, 2π), nmats)
            distance ~ filldist(Uniform(10, 50), nmats)
            position = cat(to_cartesian.(angle, distance)...; dims=2)
            position = cat(zeros(2), position; dims=2)
            speed ~ Uniform(10, 100)
            # NOTE Towards the first is preferred to make sure entities meet
            velocity = to_cartesian.(first(angle), speed) # towards the first
            #velocity = to_cartesian.(mean(angle), speed) # towards the middle
            velocity = cat(velocity, fill(zeros(2), nmats)...; dims=2)
            state0 = State(position=position, velocity=velocity)
            traj = simulate(entity, state0, scenario.dt, scenario.nsteps, getforce)
            Scene(entity=entity, traj=traj)
        end
        return Scenario(scenes=scenes, dt=scenario.dt, nsteps=scenario.nsteps), attribute
    else
        scenario = newby(scenario) do entity, group
            entity.friction[2:end] .= friction
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end
