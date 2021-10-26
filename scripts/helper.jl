function generate_and_save(generator, dir)
    entity, state0, traj, positions, velocitys, p̃ositions = generator()
    wsave("$dir/sim.bson", @dict(entity, state0, traj, positions, velocitys, p̃ositions))
    open("$dir/info.txt", "w") do io
        foreach(1:length(entity)) do i
            println(io, "# Entity $i")
            println(io, entity[i])
        end 
    end
    for (suffix, τ) in zip(
        ["", "-noisy"], [traj, map(p -> State(position=p), p̃ositions)]
    )
        p = visualize(entity, state0, τ)
        savefig(p, "$dir/traj$suffix.png")
        anim = visualize(entity, state0, τ; anim=true)
        gif(anim, "$dir/traj$suffix.gif"; show_msg=false)
    end
end
