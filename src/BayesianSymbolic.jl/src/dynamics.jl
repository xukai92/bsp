abstract type AbstractDynamics end

run(dyn::AbstractDynamics, entity::Entity, args...; kwargs...) = 
    run(dyn, getforce, entity, args...; kwargs...)

function solve end

###

struct Euler{T<:AbstractFloat} <: AbstractDynamics 
    dt::T
end

# KILL Make all constraint solvers to work with vectorization

solve(euler::Euler, entity::Entity, state::State, ::Nothing, f_e; kwargs...) = 
    zeros(size(state.velocity)...)

function solve(euler::Euler, entity::Entity, state::State, dc::DistanceConstraint, f_e; β=0.2)
    @unpack dt = euler
    @unpack mass = entity
    @unpack position, velocity = state
    @unpack center, radius = dc
    Δpos = position - center
    Δpos² = dot(Δpos, Δpos)
    b = β * -0.5 * (Δpos² - radius^2) / dt
    λ = (b - dot(Δpos, velocity + dt ./ mass' .* f_e)) ./ (dt ./ mass' * Δpos²)
    return λ .* Δpos
end

function cal_q(cc::ContactConstraint, p)
    @unpack normal = cc
    λ = p' * normal ./ -dot(normal, normal)
    return λ, p + λ .* normal
end

function solve(euler::Euler, entity::Entity, state::State, cc::ContactConstraint, f_e; kwargs...)
    @unpack position, velocity = state
    @unpack normal, level = cc
    _, q₀ = cal_q(cc, position)
    q = q₀ + normal / norm(normal) * level
    δ = position - q
    distance = norm(δ)
    is_satisfied = dot(δ, normal) > 0
    if is_satisfied || distance < 1e-3
        return zeros(size(velocity)...)
    else
        return solve(euler, entity, state, DistanceConstraint(q, 0), f_e; β=1.0)
    end
end

function cal_q(sc::SegmantConstraint, p)
    @unpack a, b = sc
    δab = b - a
    λ = dot(p - a, δab) / dot(δab, δab)
    return λ, a + λ * δab
end

function solve(euler::Euler, entity::Entity, state::State, sc::SegmantConstraint, f_e; kwargs...)
    @unpack position, velocity = state
    λ, q = cal_q(sc, position)
    if 0 <= λ <= 1
        _, q₀ = cal_q(sc, [0, 0])
        normal = position - q
        if xor(sc.isup, sign(normal[2]) == 1)
            normal *= -1
        end
        level = norm(q₀) * sign(q₀[2])
        return solve(euler, entity, state, ContactConstraint(normal, level), f_e; kwargs...)
    else
        return zeros(size(velocity)...)
    end
end

function run(
    euler::Euler, getforce::Function, entity::Entity, state::State, constraint::T=nothing
) where {T}
    @unpack dt = euler
    @unpack dynamic, mass = entity
    @unpack position, velocity = state
    force_ext = getforce(entity, state)
    force_con = solve(euler, entity, state, constraint, force_ext)
    force = force_ext .+ force_con
    velocity += dt * (force ./ mass' .* dynamic')
    position += dt * (velocity .* dynamic')
    return State(position=position, velocity=velocity)
end

###

struct Leapfrog{T<:AbstractFloat} <: AbstractDynamics 
    dt::T
end

function run(
    leapfrog::Leapfrog, getforce::Function, entity::Entity, state::State, constraint=nothing
)
    @unpack dt = leapfrog
    @unpack dynamic, mass = entity
    @unpack position, velocity = state
    # Use cached force whenever avaiable
    force = typeof(state.force) <: Nothing ? getforce(entity, state) : state.force
    velocity += dt / 2 * (force ./ mass' .* dynamic')
    position += dt * (velocity .* dynamic')
    force = getforce(entity, State(position=position, velocity=velocity))
    velocity += dt / 2 * (force ./ mass' .* dynamic')
    return State(position=position, velocity=velocity, force=force)
end

###

struct Aristotle{T<:AbstractFloat} <: AbstractDynamics 
    dt::T
    η::T
end

function run(
    aristotle::Aristotle, getforce::Function, entity::Entity, state::State, constraint::T=nothing
) where {T}
    @unpack dt, η = aristotle
    # Step 1: Run Euler's method as usual
    state = run(Euler(dt), getforce, entity, state, constraint)
    @unpack position, velocity = state
    # Step 2: Decay velocity to mimic "motion requires force"
    return State(position=position, velocity=(η * velocity))
end

### Deprecation

@deprecate run_dynamics(getforce::Function, entity::Entity, state::State, dt::AbstractFloat) run(Leapfrog(dt), getforce, entity, state)
@deprecate run_dynamics(entity::Entity, state::State, dt::AbstractFloat) run(Leapfrog(dt), getforce, entity, state)