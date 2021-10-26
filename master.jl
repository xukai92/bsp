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

@cast function efficiency(dataset::String; nobsp::Bool=false)
    for ntrains in 1:10
        # BSP
        !nobsp && Threads.@threads for (seed, shuffleseed) in cproduct(1:3, 1:5)
            `julia scripts/runexp.jl monly $dataset $ntrains --seed=$seed --shuffleseed=$shuffleseed --slient` |> try_run
        end
        # Neural baselines
        for model in ["ogn", "in", "mlpforce", "mlpdynamics"]
            Threads.@threads for (seed, shuffleseed) in cproduct(1:3, 1:5)
                `julia scripts/runexp.jl monly_neural $dataset $ntrains $model --seed=$seed --shuffleseed=$shuffleseed --slient --logging` |> try_run
            end
        end
    end
end

@cast function ablation(dataset::String; nofull::Bool=false)
    Threads.@threads for (ntrains, seed, shuffleseed, priordim, priortrans) in cproduct(1:10, 1:3, 1:5, [false, true], [false, true])
        nofull && (priordim && priortrans) && continue
        `julia scripts/runexp.jl monly $dataset $ntrains --seed=$seed --shuffleseed=$shuffleseed --priordim=$priordim --priortrans=$priortrans --slient` |> try_run
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

##### Deprecated

# Threads.@threads for (seed, shuffleseed) in cproduct(1:3, 1:6)
#     `julia scripts/runexp.jl em phys101/spring 2 1 --seed=$seed --shuffleseed=$shuffleseed --slient --logging` |> try_run
# end

# Threads.@threads for (ntrains, seed, shuffleseed) in cproduct(1:3, 0:4, -1:1)
#     `julia scripts/runexp.jl em synth/mat $ntrains 20 10 --seed=$seed --shuffleseed=$shuffleseed --weak --logging --slient` |> try_run
# end

# Threads.@threads for (ntrains, seed, shuffleseed) in cproduct(1:5, 0:9, -1:3)
#     `julia scripts/runexp.jl em synth/mat $ntrains 10 --seed=$seed --shuffleseed=$shuffleseed --slient --weak --logging` |> try_run
#     `julia scripts/runexp.jl em synth/mat $ntrains 10 --seed=$seed --shuffleseed=$shuffleseed --slient --logging` |> run_log
# end

# Map _ to - in previous generated results
#let (root, dirs, files) = first(walkdir("data/revamp"))
#    for file in files
#        src = joinpath(root, file)
#        dst = joinpath(root, replace(file, "_" => "-"))
#        println("Move $src to $dst")
#        mv(src, dst)
#    end
#end
