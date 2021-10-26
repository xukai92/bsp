using Random: AbstractRNG
using Distributions: DiscreteMultivariateDistribution
import Distributions: rand, logpdf

const G_NBODY = 8167.007

function get_speed(mass, mass′, r)
    return sqrt(abs(G_NBODY * mass * mass′) / (mass * r))
end

function get_velocity(mass, mass′, pos, pos′)
    Δpos = pos′ - pos
    speed = get_speed(mass, mass′, norm(Δpos))
    a = [-Δpos[2], Δpos[1]]
    return a / norm(a) * speed
end

struct OrbitVelocity <: DiscreteMultivariateDistribution
    mass
    mass′
    pos
    pos′
end

Base.length(::OrbitVelocity) = 2

function rand(rng::AbstractRNG, ov::OrbitVelocity)
    @unpack mass, mass′, pos, pos′ = ov
    return rand(rng, [-1, 1]) * get_velocity(mass, mass′, pos, pos′)
end

function logpdf(ov::OrbitVelocity, x::AbstractVector)
    @unpack mass, mass′, pos, pos′ = ov
    vel = get_velocity(mass, mass′, pos, pos′)
    if x == vel || x == -vel
        return log(0.5)
    else
        return log(0)
    end
end

function to_cartesian(angle, distance)
    x, y = cos(angle), sin(angle)
    return distance * [x, y]
end

@model function NBodyScenario(
    scenario, attribute, getforce, like; noprior=false, nbodies=missing
)
    if ismissing(nbodies)
        nbodies = length(first(scenario.scenes).entity)
    end
    if noprior
        @assert !ismissing(attribute)
    else
        prior_attribute = filldist(Uniform(2e-2, 9), nbodies - 1)
        attribute ~ prior_attribute
    end
    mass = [1e3, attribute...]
    if eltype(scenario.scenes) <: Missing
        entity = Entity(dynamic=fill(true, nbodies), mass=mass, shape=fill(Disc(1e0), nbodies))
        scenes = map(1:length(scenario.scenes)) do i
            angle ~ filldist(Uniform(0, π), nbodies - 1)
            distance ~ filldist(Uniform(10, 100), nbodies - 1)
            position = cat(zeros(2), to_cartesian.(angle, distance)...; dims=2)
            velocity ~ arraydist([OrbitVelocity(mass[i], mass[1], position[:,i], position[:,1]) for i in 2:nbodies])
            dv ~ filldist(filldist(Uniform(-20, 20), 2), nbodies)
            state0 = State(position=position, velocity=(hcat(zeros(2), velocity) + dv))
            traj = simulate(entity, state0, scenario.dt, scenario.nsteps, getforce)
            Scene(entity=entity, traj=traj)
        end
        return Scenario(scenes=scenes, dt=scenario.dt, nsteps=scenario.nsteps), attribute
    else
        scenario = newby(scenario) do entity, group
            entity.mass .= mass
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end
