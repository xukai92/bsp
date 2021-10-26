@model function BounceScenario(scenario, attribute, getforce, like; noprior=false, ndiscs=missing)
    if ismissing(ndiscs)
        ndiscs = length(first(scenario.scenes).entity) - 4
    end
    if noprior
        @assert !ismissing(attribute)
    else
        prior_attribute = filldist(Uniform(2e-2, 9), ndiscs)
        attribute ~ prior_attribute
    end
    mass = attribute
    if eltype(scenario.scenes) <: Missing
        dynamic = fill(true, ndiscs)
        shape = fill(Disc(1e1), ndiscs)
        dynamic = [dynamic..., false, false, false, false]
        mass = [mass..., fill(1e3, 4)...]
        depth = 50
        shape = [
            shape...,
            RectWall(200 + depth, depth), RectWall(200 + depth, depth),
            RectWall(depth, 200 + depth), RectWall(depth, 200 + depth),
        ]
        entity = Entity(dynamic=dynamic, mass=mass, shape=shape)

        scenes = map(1:length(scenario.scenes)) do idx
            local state0
            while true
                angle_position = rand(filldist(Uniform(0, 2π), ndiscs))
                distance = rand(filldist(Uniform(5, 50), ndiscs))
                position = cat(to_cartesian.(angle_position, distance)...; dims=2)
                angle_velocity = rand(filldist(Uniform(0, 2π), ndiscs))
                speed = rand(filldist(Uniform(60, 100), ndiscs))
                velocity = cat(to_cartesian.(angle_velocity, speed)...; dims=2)
                position = cat(position, [0, 100], [0, -100], [100, 0], [-100, 0]; dims=2)
                velocity = cat(velocity, fill(zeros(2), 4)...; dims=2)
                state0 = State(position=position, velocity=velocity)
                # Ensure no collison for state 0
                no_collision = true
                for i in 1:ndiscs, j in 1:ndiscs
                    if i != j
                        no_collision &= !BayesianSymbolic.does_collide(
                            entity[i], state0[i].position, entity[j], state0[j].position
                        )
                    end
                end
                if no_collision
                    break
                end
            end
            traj = simulate(entity, state0, scenario.dt, scenario.nsteps, getforce)
            Scene(entity=entity, traj=traj)
        end
        return Scenario(scenes=scenes, dt=scenario.dt, nsteps=scenario.nsteps), attribute
    else
        scenario = newby(scenario) do entity, group
            entity.mass[1:4] .= mass
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end
