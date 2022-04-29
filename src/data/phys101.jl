using NPZ, Combinatorics

# NOTE Data has 30 FPS
# FIXME What is dt here exactly???
const PHYS101_DT = 1 / 10

# Fall
const PHYS101_FALL_CONFIGS = [
    ("w_block_05-table-0$i-Kinect_RGB_1-trimmed", 110.3) for i in 1:4
]

function process_fall(data, mass; aligndisc=true, verbose=false)
    # Rescale raw data
    data = (disc=data.disc / -100,)

    # Entities
    disc = Entity(dynamic=true, mass=(mass / 1e3), shape=Disc(2e-1))
    table = Entity(dynamic=false, mass=1e1, shape=RectWall(10, 1))
    entity = cat(disc, table)

    # Traj
    xaligned = vec(mean(data.disc; dims=1))[1]
    pstable = [xaligned, -5]
    # Optionally (depending on aligndisc) align the x-axis for disc
    # This is needed as the actual data sometimes contains rigde body ratation,
    # which we want to remove from the trianing signal as BSP doesn't handle
    # this type of things yet.
    ps = map(i -> hcat(aligndisc ? [xaligned, data.disc[i,2]] : data.disc[i,:], pstable), 1:size(data.disc, 1))
    traj = collect(map(p -> State(position=p), ps))

    # Pre-processing
    smooth!(traj; eps=PHYS101_DT, k=2, q=0.999, verbose=false)

    return entity, traj
end

function loadfall(dir, idcs; verbose=false, kwargs...)
   return map(PHYS101_FALL_CONFIGS[idcs]) do (dataref, mass)
        data = (disc=npzread("$dir/$dataref.npy"),)
        verbose && @info "Raw data" size(data.disc)

        entity, traj = process_fall(data, mass; verbose=verbose, kwargs...)
        scenes = [Scene(entity=entity, traj=traj)]
        Scenario(scenes=scenes, dt=PHYS101_DT, nsteps=length(traj))
    end, fill(missing, length(idcs))
end

# Spring
#const PHYS101_SPRING_CONFIGS = [
#    ("w_block_05-tight-01-Kinect_RGB_1", 25, 70, 110.3),
#    ("w_block_05-tight-04-Kinect_RGB_1", 25, 70, 110.3),
#    ("w_block_10-tight-01-Kinect_RGB_1", 35, 80, 230.8),
#    ("w_block_10-tight-02-Kinect_RGB_1", 35, 80, 230.8),
#]
const PHYS101_SPRING_CONFIGS = [
    # filename, start_frame, end_frame, disc_mass
    ("v1", 20, 180, 110.3),
    ("v2", 10, 100, 110.3),
    ("v3", 10, 100, 230.8),
]

function process_spring(data, fstart, fend, mass; verbose=false)
    # Cut and rescale raw data
    @assert size(data.disc, 1) == size(data.spring, 1)
    data = (
        disc = data.disc[fstart:fend,:] / -100,
        spring = data.spring[fstart:fend,:] / -100,
    )

    # Entities
    disc = Entity(dynamic=true, mass=(mass / 1e3), shape=Disc(2e-1))
    spring = Entity(dynamic=false, mass=1e1, shape=Spring(missing, missing, disc))
    entity = cat(disc, spring)

    # Traj
    pspring = vec(mean(data.spring; dims=1))
    ps = map(i -> hcat(data.disc[i,:], pspring), 1:size(data.disc, 1))
    traj = collect(map(p -> State(position=p), ps))

    # Pre-processing
    smooth!(traj; eps=PHYS101_DT, k=2, q=0.999, verbose=false)

    return entity, traj
end

function loadspring(dir, idcs; verbose=false, kwargs...)
   scenarios = map(PHYS101_SPRING_CONFIGS[idcs]) do (dataref, fstart, fend, mass)
        data = (disc=npzread("$dir/$dataref-disc.npy"), spring=npzread("$dir/$dataref-spring.npy"))
        verbose && @info "Raw data" size(data.disc) size(data.spring)

        entity, traj = process_spring(data, fstart, fend, mass; verbose=verbose)
        scenes = [Scene(entity=entity, traj=traj)]
        scenario = Scenario(scenes=scenes, dt=PHYS101_DT, nsteps=length(traj))

        scenario = update_attribute(scenario, missing, missing)
    end
    return lengthfit(scenarios; slient=!verbose), fill(missing, length(idcs))
end

function update_attribute(scenario, t, l=nothing)
    scene = scenario.scenes[1]
    entity = scene.entity
    shape = entity.shape
    entity = @set(entity.shape = [shape[1], Spring(t, (isnothing(l) ? shape[2].length : l), shape[2].target)])
    scene = @set(scene.entity = entity)
    return @set(scenario.scenes = [scene])
end

# NOTE This function is not currently used.
function estimate_length(traj; method=:minmax)
    @assert method in (:minmax, :derivative)
    Pdisc = hcat(map(t -> t.position[:,1], traj)...)
    Pspring = hcat(map(t -> t.position[:,2], traj)...)
    # FIXME This method is wrong as it doesn't consider gravity.
    #       We should instead use the relation between amplitude and
    #       total energy to figure out what's the correct length.
    if method == :minmax
        ydisc = (minimum(Pdisc[2,:]) + maximum(Pdisc[2,:])) / 2
        yspring = mean(Pspring[2,:])
        l = abs(ydisc - yspring)
    end
    # TODO Finish this implementation
    if method == :derivative
        # 1. Get the second derivative
        # 2. Get the index for zeros
        # 3. Return the corresponding position
    end
    return l
end

function lengthfit(scenarios, getforce=getforce_spring; method=:bayesian, slient=false)
    @assert method in (:bayesian, :heuristic)
    @unpack ealg, elike = loadpreset_spring()
    ealg = @set(ealg.rsamples=0)
    latents = estep(ealg, SpringScenarioUnkownLength, scenarios, getforce, elike; verbose=!slient)
    latents_est = expect.(x -> x, latents)
    return map(enumerate(scenarios)) do (i, scenario)
        _, l = latents_est[i]
        if method == :heuristic
            l = estimate_length(scenario.scenes[1].traj)
        end
        update_attribute(scenario, 0.0, l)
    end
end

function tensorfit(scenarios, getforce=getforce_spring; slient=false)
    @unpack ealg, elike = loadpreset_spring()
    ealg = @set(ealg.rsamples=0)
    latents = estep(ealg, SpringScenario, scenarios, getforce, elike; verbose=!slient)
    latents_est = expect.(x -> x, latents)
    return map(enumerate(scenarios)) do (i, scenario)
        t, = latents_est[i]
        update_attribute(scenario, t)
    end
end
