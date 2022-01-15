using JSON, LinearAlgebra

const ULLMAN_DT = 0.0125

const COLOR_MAP = Dict(
    # Circle
    "red_circle" => :red, "yellow_circle" => :yellow, "blue_circle" => :blue,
    # Rectangle
    "green_rectangle" => :green, "purple_rectangle" => :purple, "brown_rectangle" => :brown
)

const COLOR2IDX = Dict(k => v for (v, k) in enumerate(values(COLOR_MAP)))
const IDX2COLOR = Dict(v => k for (v, k) in enumerate(values(COLOR_MAP)))

const MASS_MAP_ALL = Dict(
    1 => Dict("red_circle" => 3e0, "yellow_circle" => 1e0, "blue_circle" => 9e0),
    2 => Dict("red_circle" => 9e0, "yellow_circle" => 3e0, "blue_circle" => 1e0),
    3 => Dict("red_circle" => 1e0, "yellow_circle" => 9e0, "blue_circle" => 3e0),
    4 => Dict("red_circle" => 3e0, "yellow_circle" => 1e0, "blue_circle" => 9e0),
    5 => Dict("red_circle" => 9e0, "yellow_circle" => 3e0, "blue_circle" => 1e0),
    6 => Dict("red_circle" => 1e0, "yellow_circle" => 9e0, "blue_circle" => 3e0),
    7 => Dict("red_circle" => 1e0, "yellow_circle" => 9e0, "blue_circle" => 3e0),
    8 => Dict("red_circle" => 3e0, "yellow_circle" => 1e0, "blue_circle" => 9e0),
    9 => Dict("red_circle" => 9e0, "yellow_circle" => 3e0, "blue_circle" => 1e0),
   10 => Dict("red_circle" => 3e0, "yellow_circle" => 1e0, "blue_circle" => 9e0),
)
const CHARGE_MAP_ALL = Dict(
    1 => Dict("red_circle" => 0, "yellow_circle" => 0, "blue_circle" => 0),
    2 => Dict("red_circle" => 1, "yellow_circle" => 0, "blue_circle" => -1),
    3 => Dict("red_circle" => 1, "yellow_circle" => -1, "blue_circle" => 0),
    4 => Dict("red_circle" => 1, "yellow_circle" => -1, "blue_circle" => 0),
    5 => Dict("red_circle" => 1, "yellow_circle" => 0, "blue_circle" => -1),
    6 => Dict("red_circle" => 0, "yellow_circle" => 1, "blue_circle" => 1),
    7 => Dict("red_circle" => 0, "yellow_circle" => 1, "blue_circle" => -1),
    8 => Dict("red_circle" => 0, "yellow_circle" => 1, "blue_circle" => -1),
    9 => Dict("red_circle" => 1, "yellow_circle" => 0, "blue_circle" => -1),
   10 => Dict("red_circle" => 1, "yellow_circle" => -1, "blue_circle" => 0),
)
const FRICTION_MAP_ALL = Dict(
    1 => Dict("green_rectangle" => 2e1, "purple_rectangle" => 0e0, "brown_rectangle" => 5e0),
    2 => Dict("green_rectangle" => 0e0, "purple_rectangle" => 5e0, "brown_rectangle" => 2e1),
    3 => Dict("green_rectangle" => 5e0, "purple_rectangle" => 2e1, "brown_rectangle" => 0e0),
    4 => Dict("green_rectangle" => 2e1, "purple_rectangle" => 0e0, "brown_rectangle" => 5e0),
    5 => Dict("green_rectangle" => 2e1, "purple_rectangle" => 0e0, "brown_rectangle" => 5e0),
    6 => Dict("green_rectangle" => 0e0, "purple_rectangle" => 5e0, "brown_rectangle" => 2e1),
    7 => Dict("green_rectangle" => 0e0, "purple_rectangle" => 5e0, "brown_rectangle" => 2e1),
    8 => Dict("green_rectangle" => 5e0, "purple_rectangle" => 2e1, "brown_rectangle" => 0e0),
    9 => Dict("green_rectangle" => 5e0, "purple_rectangle" => 2e1, "brown_rectangle" => 0e0),
   10 => Dict("green_rectangle" => 2e1, "purple_rectangle" => 0e0, "brown_rectangle" => 5e0),
)

const COLOR2TYPE = Dict(
    :red => :circle, :yellow => :circle, :blue => :circle,
    :green => :rectangle, :purple => :rectangle, :brown => :rectangle,
)

const COLOR2LOCALIDX = Dict(
    :red => 1, :yellow => 2, :blue => 3,
    :green => 1, :purple => 2, :brown => 3,
)

function idx2type(idx)
    if idx in keys(IDX2COLOR)
        return COLOR2TYPE[IDX2COLOR[idx]]
    else
        return :wall
    end
end

# Primitives

struct Rectangle{T<:AbstractArray{<:Int}, F<:Union{AbstractFloat, Int}}
    position::T
    width::F
    height::F
end

function Rectangle(vertex_and_size)
    position = Vector{Int}(vertex_and_size[1:2])
    width, height = vertex_and_size[3:4]
    return Rectangle(position, width, height)
end

@recipe function f(rectangle::Rectangle)
    @unpack position, width, height = rectangle
    vertices = Array{Array{Int}}([position, position + [width, 0], position + [width, height], position + [0, height]])
    @series begin
        map(i -> vertices[i][1], [1,2,3,4,1]), 480 .- map(i -> vertices[i][2], [1,2,3,4,1])
    end
    return nothing
end

# Ullman

struct UllmanWorld
    static
    dynamic
end

function UllmanWorld(fpath::String)
    world_dict = JSON.parse(open(fpath))
    return UllmanWorld(world_dict["static"], world_dict["dynamic"])
end

Base.length(world::UllmanWorld) = length(world.dynamic)

@recipe function f(world::UllmanWorld, frame_idx::Int)
    @unpack static, dynamic = world
    xlims := (0, 640)
    ylims := (0, 480)
    for (rectangle, metas) in static
        color = COLOR_MAP[rectangle]
        for vertex_and_size in metas
            @series begin
                seriescolor := color
                fill := (0, 0.5, color)
                label := rectangle
                Rectangle(vertex_and_size)
            end
        end
    end
    for (circle, locations) in dynamic[frame_idx]
        for (count, loc) in enumerate(locations)
            @series begin
                seriestype := :scatter
                markersize := 40
                seriescolor := COLOR_MAP[circle]
                markerstrokewidth := 3
                markerstrokecolor := :black
                label := "$(circle)_$count"
                [loc[1] + 40], [480 - loc[2] - 40]
            end
        end
    end
    return nothing
end

l2dist(v) = sqrt(dot(v, v))

"""
Match order by L2 distances. This function will also resolve potential
mismatches of the number of entities by copying the previous state.
"""
function match_order(states_1, states_2)
    if length(states_1) > length(states_2)
        states_1, states_2 = states_2, states_1
    end
    states = [state for state in states_2]
    states_2_with_idx = collect(enumerate(states_2))
    states_2_with_idx_sorted = []
    for state1 in states_1
        ds = [l2dist(state1.position - state2.position) for (_, state2) in states_2_with_idx]
        i = argmin(ds)
        push!(states_2_with_idx_sorted, states_2_with_idx[i])
        popat!(states_2_with_idx, i)
    end
    for (i, (j, _)) in enumerate(states_2_with_idx_sorted)
        states[j] = states_1[i]
    end
    return states
end

function make_walls(width=480, height=640, depth=95)
    shapes = [
        RectWall(height + depth, depth), RectWall(height + depth, depth),
        RectWall(depth, width + depth), RectWall(depth, width + depth),
    ]
    positions = [[height/2, -depth/4], [height/2, width+depth/4], [-depth/4, width/2], [height+depth/4, width/2]]
    return map(shape -> Entity(dynamic=false, mass=1e3, shape=shape), shapes), map(pos -> State(position=pos), positions)
end

function Main.parse(world::UllmanWorld, mass_map, charge_map, friction_map)
    @unpack static, dynamic = world
    entities = []
    colors = []
    states_static = []
    states_dynamic_list = []
    for (rectangle, metas) in static
        color = COLOR_MAP[rectangle]
        friction = friction_map[rectangle]
        for vertex_and_size in metas
            rectangle = Rectangle(vertex_and_size)
            entity = Entity(dynamic=false, mass=1e0, friction=friction, shape=RectMat(rectangle.width, rectangle.height))
            push!(entities, entity)
            push!(colors, color)
            state = State(position=Float64[rectangle.position[1] + rectangle.width / 2, 480 - rectangle.position[2] - rectangle.height / 2])
            push!(states_static, state)
        end
    end
    first_iteration = true
    for frame_idx in 1:length(world)
        states_dynamic = []
        for (circle, locations) in dynamic[frame_idx]
            color = COLOR_MAP[circle]
            mass = mass_map[circle]
            charge = charge_map[circle]
            for (count, position) in enumerate(locations)
                if first_iteration
                    entity = Entity(dynamic=true, mass=mass, charge=charge, shape=Disc(4e1 + 1e0))
                    push!(entities, entity)
                    push!(colors, color)
                end
                state = State(position=Float64[position[1] + 40, 480 - position[2] - 40])
                push!(states_dynamic, state)
            end
        end
        if !first_iteration
            states_dynamic = match_order(states_dynamic, states_dynamic_list[end])
        end
        first_iteration = false
        push!(states_dynamic_list, states_dynamic)
    end
    # Avoid the case that the first one is mis-counted
    states_dynamic_list[1] = match_order(states_dynamic_list[1], states_dynamic_list[end])
    num_dynamic = length(first(states_dynamic_list))
    states_dynamic_list_filtered = filter(states -> length(states) == num_dynamic, states_dynamic_list)
    num_frames_filtered = length(states_dynamic_list) - length(states_dynamic_list_filtered)
    if !iszero(num_frames_filtered)
        println("$num_frames_filtered incomplete frame(s) removed.")
    end
    entities_wall, states_wall = make_walls()
    group_map = map(c -> COLOR2IDX[c], colors)
    group_map = [group_map..., fill(length(COLOR2IDX) + 1, 4)...]
    return cat(entities..., entities_wall...), map(states -> cat(states_static..., states..., states_wall...), states_dynamic_list_filtered[2:end]), group_map
end

# Loading

function process_ullman(entity, traj, k, q, discard, thinning; regenbsp=false, verbose=false)
    traj = deepcopy(traj)

    smooth!(traj; eps=ULLMAN_DT, k=k, q=q, verbose=verbose)

    # TODO Update this branch with the new constants
    if regenbsp
        getforce = (e, s) -> getforce_pairwise(e, s; C=27.3, G₀=4.85, G=0.998)

        V = makeV(traj; eps=ULLMAN_DT)
        foreach(enumerate(traj)) do (k, state)
            if k > 1
                state.velocity .= V[:,:,k-1]
            end
        end

        traj = traj[discard:thinning:end-discard]

        T, traj = length(traj), [traj[1]]
        for t in 1:T-1
            push!(traj, run(Euler(ULLMAN_DT * thinning), getforce, entity, traj[end]))
        end
    else
        traj = traj[discard:thinning:end-discard]
    end

    return entity, traj
end

function loadullman(dir, wid; idcs=1:6, thinning=2, regen=false, verbose=false)
    mass_map, charge_map, friction_map = MASS_MAP_ALL[wid], CHARGE_MAP_ALL[wid], FRICTION_MAP_ALL[wid]
    scenes = map(idcs) do i
        world = UllmanWorld("$dir/world$(wid)_$i.json")

        entity, traj, group_map = parse(world, mass_map, charge_map, friction_map)

        entity, traj = process_ullman(entity, traj, 2, 0.999, 5, thinning; regenbsp=regen, verbose=verbose)

        Scene(entity=entity, traj=traj, group=group_map)
    end

    scenario = Scenario(scenes=scenes, dt=(ULLMAN_DT * thinning), nsteps=length(scenes[1].traj))
    attribute = Int[values(mass_map)..., values(charge_map)..., values(friction_map)...]
    return scenario, attribute
end

# Visualization

function generate_gif(scenedix; suffix=nothing)
    world = UllmanWorld(datadir("ullman", "processed"), scenedix)

    anim = @animate for frame_idx in 1:length(world)
        plot(world, frame_idx, size=(640, 480))
    end

    fn_without_ext, _ = split(scenedix, ".")
    if !isnothing(suffix)
        fn = "$(fn_without_ext)_$suffix"
    else
        fn = fn_without_ext
    end
    gif(anim, datadir("ullman", "rerendered_raw", "$fn.gif"); show_msg=false)
end

function generate_gif_bsp(scenedix; suffix=nothing)
    world = UllmanWorld(datadir("ullman", "processed"), scenedix)

    entity, traj, _ = parse(world)

    anim = @animate for t in 1:length(traj)
        plot(entity, traj, t)
    end

    fn_without_ext, _ = split(scenedix, ".")
    if !isnothing(suffix)
        fn_without_ext = "$(fn_without_ext)_$suffix"
    else
        fn = fn_without_ext
    end
    gif(anim, datadir("ullman", "rerendered", "$fn.gif"); show_msg=false)
end

function make_pos_plot(traj)
    P = makeP(traj)

    p = plot(title="pos", size=(400, 400))
    for i in 2+1:10-4
        pi = P[:,i,:]
        plot!(p, pi[1,:], pi[2,:], label="p$i")
    end

    p
end

function make_vel_plot(traj; kwargs...)
    S = makeS(traj; kwargs...)

    p = plot(title="vel", size=(600, 200))
    for i in 2+1:10-4
        plot!(p, S[:,i], label="S$i")
    end

    p
end

function make_acc_plot(traj; kwargs...)
    F = makeF(traj; kwargs...)

    p = plot(title="acc", size=(600, 200))
    for i in 2+1:10-4
        plot!(p, F[:,i], label="F$i")
    end

    p
end
