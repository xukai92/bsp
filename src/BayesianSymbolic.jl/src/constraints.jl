abstract type AbstractConstraint end

"""
C = dot(p, p) - l^2
"""
struct DistanceConstraint{CenterT, RadiusT} <: AbstractConstraint
    center::CenterT
    radius::RadiusT
end

"""
Fall back to DistanceConstraint by setting
center = projected point
radius = level
"""
struct ContactConstraint{NormalT, LevelT} <: AbstractConstraint
    normal::NormalT
    level::LevelT
end

"""
Fall back to ContactConstraint by setting
normal = normal vector of (b - a)
level = distance of [0, 0] to (b - a)
"""
struct SegmantConstraint{AT, BT, IsupT} <: AbstractConstraint
    a::AT
    b::BT
    isup::IsupT
end
