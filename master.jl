##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

using Comonicon

function try_run(cmd)
    try
        @info "Running" cmd
        run(cmd)
    catch
        @info "Fail to run" cmd
    end
end

# Collected product that has known length for @threads
cproduct(args...) = collect(Iterators.product(args...))

@cast function efficiency(dataset::String; nobsp::Bool=false, noneural::Bool=false)
    for ntrains in 1:10
        # BSP
        !nobsp && Threads.@threads for (seed, shuffleseed) in cproduct(1:3, 1:4)
            `julia scripts/runexp.jl monly $dataset $ntrains --seed=$seed --shuffleseed=$shuffleseed --slient` |> try_run
        end
        # Neural baselines
        !noneural && for model in ["ogn", "in", "mlpforce", "mlpdynamics"]
            Threads.@threads for (seed, shuffleseed) in cproduct(1:3, 1:4)
                `julia scripts/runexp.jl monly-neural $dataset $ntrains $model --seed=$seed --shuffleseed=$shuffleseed --slient --logging` |> try_run
            end
        end
    end
end

@cast function ablation(dataset::String; nofull::Bool=false)
    for ntrains in 1:10
        Threads.@threads for (seed, shuffleseed, nopriordim, nopriortrans) in cproduct(1:3, 1:4, [false, true], [false, true])
            nofull && (~nopriordim && ~nopriortrans) && continue
            nopriordimarg = nopriordim ? `--nopriordim` : ``
            nopriortransarg = nopriortrans ? `--nopriortrans` : ``
            `julia scripts/runexp.jl monly $dataset $ntrains --seed=$seed --shuffleseed=$shuffleseed $nopriordimarg $nopriortransarg --slient` |> try_run
        end
    end
end

@cast function em(dataset::String)
    Threads.@threads for (seed, shuffleseed) in cproduct(1:3, 1:5)
        `julia scripts/runexp.jl em $dataset 5 3 --seed=$seed --shuffleseed=$shuffleseed --slient --logging` |> try_run
    end
end

@cast function phys101(dataset::String)
    cmds = [
        `julia scripts/runexp.jl em phys101/$dataset 2 5 --seed=$seed --shuffleseed=$shuffleseed` for seed in 1:3, shuffleseed in 1:6
    ]
    Threads.@threads for cmd in cmds
        try_run(cmd)
    end
end

@cast function ullman(niters::Int)
    cmds = [
        `julia scripts/runexp_ullman.jl $wid $sid $niters` for wid in 1:10, sid in 1:6
    ]
    Threads.@threads for cmd in cmds
        try_run(cmd)
    end
end

@main
