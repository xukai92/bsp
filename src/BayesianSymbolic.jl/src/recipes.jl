### Helper functions

function squared_xylims(xlims, ylims)
    xspan = xlims[2] - xlims[1]
    yspan = ylims[2] - ylims[1]
    Δspan = abs(xspan - yspan)
    if xspan < yspan
        xlims = xlims .+ [-Δspan, Δspan] / 2
    else
        ylims = ylims .+ [-Δspan, Δspan] / 2
    end
    return xlims, ylims
end

function expandlims(lims, ratio)
    lim_min, lim_max = lims
    lim_diff = lim_max - lim_min
    return (lim_min - ratio * lim_diff, lim_max + ratio * lim_diff)
end

### Entity & state

"Make relative vertices."
make_relverts(entity) = make_relverts(entity.shape)

function make_relverts(shape::Disc)
    @unpack radius = shape
    return (iszero(radius) ? 0.025 : radius) * [cos.(0:0.2:2π) sin.(0:0.2:2π)]'
end

function make_relverts(shape::Union{RectMat, RectWall})
    @unpack width, height = shape
    return [width, height] .* [-1 +1 +1 -1; +1 +1 -1 -1] / 2
end

function make_relverts(shape::Spring)
    radius = 0.1
    return (iszero(radius) ? 0.025 : radius) * [cos.(0:0.2:2π) sin.(0:0.2:2π)]'
end


@recipe function f(
    entity::Entity{T}, trajectory::AbstractVector{<:State}, n_steps::Int=length(trajectory)
) where {T<:Union{Bool, AbstractVector}}
    # Data prep
    n_entities = length(entity)
    position_1_full = cat(map(t -> t.position[1,:], trajectory)...; dims=2)'
    position_2_full = cat(map(t -> t.position[2,:], trajectory)...; dims=2)'
    position_1 = position_1_full[1:n_steps,:]
    position_2 = position_2_full[1:n_steps,:]
    
    # Get extra points in vertices
    extras_1, extras_2 = [], []
    for i in 1:n_entities
        relverts = make_relverts(entity[i])
        for t in 1:size(position_1_full, 1)
            verts = relverts .+ [position_1_full[t,i], position_2_full[t,i]]
            push!(extras_1, verts[1,:]...)
            push!(extras_2, verts[2,:]...)
        end
    end

    # Plot attributes
    xlims = extrema([position_1_full..., extras_1...])
    ylims = extrema([position_2_full..., extras_2...])
    xlims, ylims = squared_xylims(xlims, ylims)
    aspect_ratio := :equal
    xlims --> expandlims(xlims, 0.1)
    ylims --> expandlims(ylims, 0.1)
    legend --> :topright
    legendfontsize --> 7

    # Entities
    extras = []
    i_dynamic = 0
    counter = Dict(T => 0 for T in [:Disc, :RectMat, :RectWall, :CircleMat, :Spring])
    for i in 1:n_entities
        if entity[i].dynamic
            color = (i_dynamic += 1)
            @series begin
                seriescolor := i
                seriesalpha := n_steps == 1 ? 1.0 : range(0, 1; length=n_steps)
                label := nothing
                position_1[:,i], position_2[:,i]
            end
        else
            color = :gray
        end
        if entity[i].shape isa RectMat
            fillalpha = 0.5
        else
            fillalpha = 1.0
        end
        relverts = make_relverts(entity[i])
        verts = relverts .+ [position_1[end,i], position_2[end,i]]
        vidcs = collect(1:size(verts, 2))

        typesymbol = nameof(typeof(entity[i].shape))
        label = "$typesymbol $(counter[typesymbol] += 1)"
        if entity[i].shape isa RectWall
            if counter[typesymbol] == 1
                label = string(typesymbol)
            else
                label = nothing
            end
        end
        @series begin
            seriescolor := color
            fill := (0, fillalpha, color)
            label := label
            verts[1,[vidcs...,1]], verts[2,[vidcs...,1]]
        end
        push!(extras, verts[1,:], verts[2,:])

        # Draw spring connection
        if entity[i].shape isa Spring
            for j in 1:n_entities
                if entity[j] == entity[i].shape.target
                    @series begin
                        seriescolor := :gray
                        linestyle := :dashdotdot
                        label := nothing
                        [position_1[end,i], position_1[end,j]], [position_2[end,i], position_2[end,j]]
                    end
                end
            end
        end
    end
    
    nothing
end

### Constraints

@recipe function f(dc::DistanceConstraint)
    @unpack center, radius = dc
    x, y = center
    @series begin
        seriestype := :scatter
        markersize --> 2
        label --> "Center"
        [x], [y]
    end
    @series begin
        seriestype := :line
        line --> :dash
        label --> "Radius"
        [x, x + radius], [y, y]
    end
    return nothing
end

@recipe function f(cc::ContactConstraint)
    @unpack normal, level = cc
    @assert prod(normal) == 0 "Only support plotting `ContactConstraint` with axis-aligned `normal`."
    x, y = normal
    line --> :dash
    label --> "Contact"
    if x == 0
        seriestype := :hline
        return [y > 0 ? level : -level]
    else
        seriestype := :vline
        return [x > 0 ? level : -level]
    end
end

@recipe function f(sc::SegmantConstraint)
    @unpack a, b = sc
    line --> :dash
    label --> "Segmant"
    return [a[1], b[1]], [a[2], b[2]]
end
