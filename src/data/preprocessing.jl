using ImageFiltering

function makeP(traj)
    cat(map(traj) do state
        state.position
    end...; dims=3)
end

function makeV(traj; eps=1)
    cat(map(1:length(traj)-1) do i
        dp = traj[i+1].position - traj[i].position
        dp / eps
    end...; dims=3)
end

function makeS(traj; kwargs...)
    V = makeV(traj; kwargs...)
    S = sqrt.(sum(V.^2; dims=1))
    dropdims(S; dims=1)'
end

function makeF(traj; eps=1)
    vcat(map(1:length(traj)-2) do i
        dp2 = (traj[i+2].position - traj[i+1].position) / eps
        dp1 = (traj[i+1].position - traj[i+0].position) / eps
        ddp = (dp2 - dp1) / eps
        sqrt.(sum(ddp.^2; dims=1))
    end...)
end

# NOTE Pre-processing the raw data
# Schematics for how three positions variables are
# used to compute one acceleration/force variable.
#  ------------
# | p1, p2, p3 |
# |     v2, v3 |
# |         a3 |
# |         f3 |
#  ------------
# p2 = p1 + v2 * dt
# p3 = p2 + v3 * dt
# v3 = v2 * a3 * dt

function smooth!(traj; eps=1, k=2, q=0.995, verbose=false)
    ker = ImageFiltering.Kernel.gaussian((k,))

    P = makeP(traj)
    Pold = copy(P)

    F = makeF(traj; eps=eps)
    threshold = quantile(vec(F), q)

    verbose && @info threshold
    mask = F .< threshold

    for i in 1:size(P, 1), j in 1:size(P, 2)
        P[i,j,:] .= imfilter(P[i,j,:], ker)
    end

    foreach(enumerate(traj)) do (k, state)
        if k > 2
            traj[k].position .= (P[:,:,k] .* mask[k-2,:]' + Pold[:,:,k] .* (1 .- mask[k-2,:]'))
        end
    end

    return F
end
