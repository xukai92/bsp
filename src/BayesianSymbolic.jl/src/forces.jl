"""
A template function to compute pairwise forces that takes an entity-level force 
function `getforce_pair`. See `getforce_pair` for an example. This template is 
useful for symbolic regression as we only aim to learn the `getforce_pair` function.

If there are N entities in total and K of them are static, the complexity is O(N * (N - K)).
"""
function getforce_pairwise(
    entity::Entity, state::State{T}, getforce_pair=getforce_pair; d=size(state, 1), kwargs...
) where {T<:Union{AbstractMatrix, AbstractArray{<:Real, 3}}}
    n = size(state, 2)
    fs = map(1:n) do i
        ei = entity[i]
        f = T <: AbstractMatrix ? zeros(d) : zeros(d, size(state, 3))
        if ei.dynamic # avoid computing force for static entity
            idcs_j = filter(j -> !(i == j), 1:n)
            if length(idcs_j) > 0
                for j in idcs_j
                    f += getforce_pair(ei, state[i], entity[j], state[j]; kwargs...)
                end
            end
        end
        T <: AbstractMatrix ? f : reshape(f, d, 1, size(f, 2))
    end
    return cat(fs...; dims=2)
end

"There is no iteration force if there is only one entity."
getforce_pairwise(::Entity, state::State{<:AbstractVector}, args...) = zeros(size(state))

function contact_point(ei, si, ej, sj)
    pi, pj = si.position, sj.position
    c = if ej.shape isa RectWall
        if ej.shape.width > ej.shape.height
            vcat(pi[1,:]', pj[2,:]')
        else
            vcat(pj[1,:]', pi[2,:]')
        end
    else
        pj
    end
    return c
end

const G = 6.67e-11
const ε = 2e-2

"An example of entity-level force function. This is part of the function we want to learn."
function getforce_pair(
    ei::Entity, si::State, ej::Entity, sj::State; C=(1 / ε), G₀=0.8, G=G, Gm=0, Gc=0
)
    pi, pj = si.position, sj.position
    massmass = ej.mass * ei.mass
    Δp = pj - pi
    rsq = sum(abs2, Δp; dims=1)
    u = Δp ./ sqrt.(rsq)
    f_total = zeros(size(si.velocity)...)
    # Collision force
    # u = (c - pi) / norm(c - pi)
    # C * 2 * mi * mj / (mi + mj) * dot(u, vj - vi) * -u * does_collide
    c = contact_point(ei, si, ej, sj)
    u_collision = (c - pi) ./ sqrt.(sum(abs2, c - pi; dims=1))
    Δv = sj.velocity - si.velocity
    coeff = 2 * massmass / (ej.mass + ei.mass)
    f_collision = coeff .* sum(abs, u_collision .* Δv; dims=1) .* -u_collision
    f_total += C * f_collision .* does_collide(ei, pi, ej, pj)
    # Friction force
    # Numerical safe normalise
    u_friction = let vi = si.velocity
        vinorm = sqrt.(sum(abs2, vi; dims=1))
        scale = ifelse.(vinorm .> 1e-9, inv.(vinorm), 0)
        -vi .* scale
    end
    Fn = G₀ * ei.mass
    Ff = Fn * ej.friction
    f_total += Ff * u_friction .* is_within(ei, pi, ej, pj)
    # Remote force
    magmag = ej.magnetism * ei.magnetism
    cc = ej.charge * ei.charge
    f_remote = (G * massmass - Gm * magmag - Gc * cc) ./ rsq .* u
    f_total += f_remote
    return f_total
end

is_within(ei::Entity, pi, ej::Entity, pj) = is_within(ei.shape, pi, ej.shape, pj)

function is_within(si::Shape, pi, sj::Shape, pj)
    if sj isa RectMat || sj isa CircleMat
        Δp = pj - pi
        if Δp isa AbstractVector
            Δx, Δy = Δp[1], Δp[2]
        else
            Δx, Δy = Δp[1,:]', Δp[2,:]'
        end
        @unpack width, height = sj
        return (-width / 2 .<= Δx .<= width / 2) .& (-height / 2 .<= Δy .<= height / 2)
    else
        return false
    end
end

does_collide(ei::Entity, pi, ej::Entity, pj) = does_collide(ei.shape, pi, ej.shape, pj)

function does_collide(si::Shape, pi, sj::Shape, pj)
    if si isa Disc && sj isa Disc
        Δp = pj - pi
        if Δp isa AbstractVector
            rsq = sum(abs2, Δp)
        else
            rsq = sum(abs2, Δp; dims=1)
        end
        return rsq .< (si.radius + sj.radius)^2
    elseif si isa Disc && sj isa RectWall
        radius = si.radius
        @unpack width, height = sj
        Δp = abs.(pj - pi)
        if Δp isa AbstractVector
            Δx, Δy = Δp[1], Δp[2]
        else
            Δx, Δy = Δp[1,:]', Δp[2,:]'
        end
        return (width / 2 + radius .> Δx) .& (height / 2 + radius .> Δy)
    else
        return false
    end
end

function getforce(entity::Entity, S::State; gravity::GT=Val(false), kwargs...) where {GT}
    f_pairwise = getforce_pairwise(entity, S; kwargs...)
    return GT <: Val{true} ? f_pairwise .+ [0, -9.8] .* entity.mass' : f_pairwise
end
