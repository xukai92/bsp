using Plots

function makeanim(scene)
    return @animate for i in 1:length(scene.traj)
        plot(scene.entity, scene.traj, i)
    end
end

makegif(scene) = gif(makeanim(scene))

function visualize(scene; anim=false)
    return if anim
        makeanim(scene)
    else
        plot(scene.entity, scene.traj)
    end
end

"Compute mean and variance of latent."
function compute_mean_variance(latent::Vector{<:WeightedSample})
    m = expect(x -> x,    latent)
    v = expect(x -> x.^2, latent) - m.^2
    return m, v
end

function compute_mean_variance(latents::Vector{<:Vector{<:WeightedSample}})
    ms = []
    vs = []
    for latent in latents
        m, v = compute_mean_variance(latent)
        push!(ms, m)
        push!(vs, v)
    end
    return ms, vs
end

# FIXME I should include an intercept variable here
function compute_coeff(attributes, latents::Vector{<:Vector{<:WeightedSample}})
    ms, vs = compute_mean_variance(latents)
    M = hcat(ms...)

    A = hcat(attributes...)

    coeff = []
    for i in 1:length(attributes[1])
        push!(coeff, (M[i:i,:] / A[i:i,:])[1])
    end
    return coeff
end

"""
`singletrace` holds the trace for a single scenario only.
"""
function make_attribute_trace_plot(attribute, singletrace; coeff=nothing, name=nothing)
    ms, vs = compute_mean_variance(singletrace::Vector{<:Vector{<:WeightedSample}})

    p = plot(size=(400, 200))
    for i in 1:length(attribute)
        namei = isnothing(name) ? "attribute[$i]" : name[i]
        hline!([attribute[i]], label="$namei (true)", color=i, alpha=0.8)
        mi = map(m -> m[i], ms)
        if !isnothing(coeff)
            mi = mi ./ coeff[i,:]
        end
        plot!(1:length(trace), mi, style=:dash, label="$namei (infer)", color=i)
    end
    return p
end

function compute_rsquared(y, f)
    tot = sum((y .- mean(y)).^2)
    res = sum((y - f).^2)
    return 1 - res / tot
end

function compute_rsquared(attributes, latents, coeff)
    # TODO Use mode instead of mean here
    ms, vs = compute_mean_variance(latents)
    fs = map(m -> m ./ coeff, ms)
    rsqs = []
    for i in 1:length(first(attributes))
        rsq = compute_rsquared(map(a -> a[i], attributes), map(f -> f[i], fs))
        push!(rsqs, rsq)
    end
    return rsqs
end

function make_rsq_plot(rsqs; name=nothing)
    p = plot(size=(400, 200))
    for i in 1:length(first(rsqs))
        namei = isnothing(name) ? "attribute[$i]" : name[i]
        plot!(1:length(rsqs), map(rsq -> rsq[i], rsqs), style=:dash, label=namei, color=i)
    end
    return p
end

function evalforce(ScenarioModel, scenarios, attributes, force::AbstractForceModel, ealg, elike, mlike)
    oracle_available = !any(ismissing.(attributes))

    latents = estep(@set(ealg.rsamples=0), ScenarioModel, scenarios, force, elike)
    rsq = oracle_available ?
        compute_rsquared(attributes, latents, compute_coeff(attributes, latents)) :
        missing
    nrmse = compute_normrmse(ScenarioModel, scenarios, latents, force, mlike)
    
    nrmse_oracle = oracle_available ?
        compute_normrmse(ScenarioModel, scenarios, make_latents(attributes), force, mlike) :
        missing
    
    return (rsq=rsq, normrmse=nrmse, normrmse_oracle=nrmse_oracle)
end

function evalm(ScenarioModel, scenarios, attributes, mmethod, mlike)
    nrmse_oracle = compute_normrmse(ScenarioModel, scenarios, make_latents(attributes), mmethod, mlike)
    return (normrmse_oracle=nrmse_oracle,)
end
