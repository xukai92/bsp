import Distributions: rand

struct MixedAttribute <: DiscreteMultivariateDistribution
    prior_mass
    prior_charge
    prior_friction
end

function MixedAttribute()
    prior_mass = arraydist(fill(DiscreteNonParametric([1, 3, 9], ones(3) / 3), 3))
    prior_charge = arraydist(fill(DiscreteNonParametric([-1, 0, 1], ones(3) / 3), 3))
    prior_friction = arraydist(fill(DiscreteNonParametric([0, 5, 20], ones(3) / 3), 3))
    return MixedAttribute(prior_mass, prior_charge, prior_friction)
end

Base.length(::MixedAttribute) = 9

function rand(rng::AbstractRNG, ma::MixedAttribute)
    return [rand(ma.prior_mass)..., rand(ma.prior_charge)..., rand(ma.prior_friction)...]
end

function logpdf(ma::MixedAttribute, x::AbstractVector)
    lp = 0
    for i in 1:3
        if x[i] in [1, 3, 9]
            lp += log(1 / 3)
        else
            return log(0)
        end
    end
    for i in 4:6
        if x[i] in [-1, 0, 1]
            lp += log(1 / 3)
        else
            return log(0)
        end
    end
    for i in 7:9
        if x[i] in [0, 5, 20]
            lp += log(1 / 3)
        else
            return log(0)
        end
    end
    return lp
end

function groupinfo(group)
    ts = idx2type.(group)
    localidcs = map(group) do idx
        if idx in keys(IDX2COLOR)
            COLOR2LOCALIDX[IDX2COLOR[idx]]
        else
            -1
        end
    end
    return (nmats=sum(ts .== :rectangle), ndiscs=sum(ts .== :circle), nwalls=sum(ts .== :wall), localidcs=localidcs)
end

@model function UllmanScenario(scenario, attribute, getforce, like; noprior=false)
    if noprior
        @assert !ismissing(attribute)
    else
        attribute ~ MixedAttribute()
    end
    if eltype(scenario.scenes) <: Missing
        @assert "FallScenario doesn't support generation."
    else
        mass, charge, friction = attribute[1:3], attribute[4:6], attribute[7:9]
        scenario = newby(scenario) do entity, group
            gi = groupinfo(group)
            for i in 1:gi.nmats
                entity.friction[1:1+i-1] .= friction[gi.localidcs[i]]
            end
            for i in 1:gi.ndiscs
                entity.mass[1+gi.nmats:1+gi.nmats+i-1] .= mass[gi.localidcs[gi.nmats+i]]
                entity.charge[1+gi.nmats:1+gi.nmats+i-1] .= charge[gi.localidcs[gi.nmats+i]]
            end
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end
