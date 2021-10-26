### Shape

abstract type Shape{T} end

struct Disc{T} <: Shape{T}
    radius::T
end

struct RectMat{T} <: Shape{T} 
    width::T
    height::T
end

struct RectWall{T} <: Shape{T} 
    width::T
    height::T
end

struct Spring{T, E} <: Shape{T}
    tensor::T
    length::T
    target::E
end

struct CircleMat{T} <: Shape{T}
    radius_x::T
    radius_y::T
end

Base.length(shape::T) where {T <: Shape}  = 0
Base.length(shape::Spring) = shape.length

Base.zero(::Type{Shape}) = Disc(zero(Float64))

### Entity

similarTf(x::AbstractArray, T=eltype(x), f=zero) = fill!(similar(x, T), f(T))
similarTf(x, T=eltype(x), f=zero) = f(T)

# NOTE Should I move shape specific attributes to shape structs?
Base.@kwdef struct Entity{
    DT<:Union{Missing, Bool, AbstractVector{Bool}}, MassT, MagT, CT, FT, ST
}
    dynamic::DT=true
    mass::MassT=similarTf(dynamic, Float64)
    magnetism::MagT=similarTf(dynamic, Float64)
    charge::CT=similarTf(dynamic, Float64)
    friction::FT=similarTf(dynamic, Float64)
    shape::ST=similarTf(dynamic, Shape)
end

function Base.cat(entity::Entity...)
    dynamic = vcat(map(e -> e.dynamic, entity)...)
    mass = vcat(map(e -> e.mass, entity)...)
    magnetism = vcat(map(e -> e.magnetism, entity)...)
    charge = vcat(map(e -> e.charge, entity)...)
    friction = vcat(map(e -> e.friction, entity)...)
    shape = vcat(map(e -> e.shape, entity)...)
    return Entity(dynamic, mass, magnetism, charge, friction, shape)
end

function Base.getindex(entity::Entity, i::Union{Int, UnitRange})
    @unpack dynamic, mass, magnetism, charge, friction, shape = entity
    return Entity(dynamic[i], mass[i], magnetism[i], charge[i], friction[i], shape[i])
end

Base.length(entity::Entity) = length(entity.dynamic)

### State

Base.@kwdef struct State{PT, VT, FT}
    position::PT
    velocity::VT=similarTf(position)
    force::FT=nothing
end

function Base.cat(state::State{T}...; dims=(T <: AbstractVector ? 2 : 3)) where {T}
    position = cat(map(e -> e.position, state)...; dims=dims)
    velocity = cat(map(e -> e.velocity, state)...; dims=dims)
    force = map(e -> e.force, state)
    force = any(isnothing, force) ? nothing : cat(force...; dims=dims)
    return State(position, velocity, force)
end

function Base.getindex(state::State{<:AbstractMatrix}, i::Union{Int, UnitRange})
    @unpack position, velocity, force = state
    return State(position[:,i], velocity[:,i], isnothing(force) ? nothing : force[:,i])
end

function Base.getindex(
    state::State{<:AbstractArray{<:Real, 3}}, 
    i::Union{Int, UnitRange, Colon}, 
    j::Union{Int, UnitRange, Colon}=Colon()
)
    @unpack position, velocity, force = state
    return State(position[:,i,j], velocity[:,i,j], isnothing(force) ? nothing : force[:,i,j])
end

Base.size(state::State, args...) = size(state.position, args...)
