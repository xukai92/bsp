# TODO Make this compatible with Ullman and Phys101 as well
function loaddata(dir, nscenarios; idcs=nothing, shuffleseed=-1, istrain=true, verbose=false)
    if occursin("synth", dir)
        if isnothing(idcs)
            # The last 20 scenarios are reserved as testing data
            idcs = istrain ? (1:80) : (81:100)
            if shuffleseed >= 0
                idcs = shuffle(MersenneTwister(shuffleseed), idcs)
            end
            idcs = idcs[1:nscenarios]
        #else
            #@assert nscenarios == length(idcs)
        end
        scenarios = []
        attributes = []
        for i in idcs
            data = wload(joinpath(dir, lpad(i, 3, "0"), "data.jld2"))
            @unpack scenario, attribute = data
            push!(scenarios, scenario)
            push!(attributes, attribute)
        end
        verbose && @info "Loaded data from $dir" idcs dict2ntuple(Dict("scenario_$i" => a for (i, a) in enumerate(attributes)))...
    elseif occursin("phys101", dir)
        scenarios, attributes = loaddata_phys101(
            dir, nscenarios; idcs=idcs, shuffleseed=shuffleseed, istrain=istrain, verbose=verbose
        )
    else
        @error "Unknown dataset $dir"
    end
    return scenarios, attributes
end

const PHYS101_IDCS_ALL = [
    [1, 2, 3, 4],
    [1, 3, 2, 4],
    [1, 4, 2, 3],
    [2, 3, 1, 4],
    [2, 4, 1, 3],
    [3, 4, 1, 2],
]

function loaddata_phys101(dir, nscenarios; idcs=nothing, shuffleseed=-1, istrain=true, verbose=false, kwargs...)
    @assert 1 <= nscenarios <= 2
    @assert shuffleseed == -1 || 1 <= shuffleseed <= 6
    if isnothing(idcs)
        # Deal with pseudo shuffling via permutation
        if shuffleseed >= 1
            idcs = PHYS101_IDCS_ALL[shuffleseed]
            idcs_train, idcs_test = idcs[1:2], idcs[3:4]
        else
            idcs_train, idcs_test = [1, 2], [3, 4]
        end
        idcs = (istrain ? idcs_train : idcs_test)[1:nscenarios]
        verbose && @info "Indices" idcs
    else
        @assert nscenarios == length(idcs)
    end
    loadfunc = if occursin("fall", dir)
        loadfall
    elseif occursin("spring", dir)
        loadspring
    end
    return loadfunc(dir, idcs; verbose=verbose)
end

include("data/preprocessing.jl")
include("data/phys101.jl")
include("data/ullman.jl")
