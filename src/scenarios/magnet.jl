@model function MagnetScenario(scenario, attribute, getforce, like)
    prior_attribute = arraydist([
        DiscreteNonParametric([i/4 for i in 1:4], ones(4) / 4),
        DiscreteNonParametric([i/8 for i in 1:8], ones(8) / 8),
    ])
    attribute ~ prior_attribute
    fric2, magn3 = attribute
    if eltype(scenario.scenes) <: Missing
        e1 = Entity(dynamic=false, mass=1e0, magnetism=-1e1, shape=Disc(2e-1))
        e2 = Entity(dynamic=false, friction=fric2, shape=RectMat(3e0, 2e0))
        e3 = Entity(dynamic=true,  mass=1e0, magnetism=magn3, shape=Disc(2e-1))
        entity = cat(e1, e2, e3)
        scenes = map(1:length(scenario.scenes)) do i
            s1 = State(position=[+2e0, +1e0])
            s2 = State(position=[+1e0, +2e0])
            velocity ~ MvNormal([+1e0, +2e0], 0.5)
            s3 = State(position=[-1e0, +0e0], velocity=velocity)
            state0 = cat(s1, s2, s3)
            traj = simulate(entity, state0, scenario.dt, scenario.nsteps, getforce)
            Scene(entity=entity, traj=traj)
        end
        return Scenario(scenes=scenes, dt=scenario.dt, nsteps=scenario.nsteps), attribute
    else
        scenario = newby(scenario) do entity, group
            entity.friction[2] = fric2
            entity.magnetism[3] = magn3
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end
