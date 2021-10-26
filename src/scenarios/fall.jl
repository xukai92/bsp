@model function FallScenario(scenario, attribute, getforce, like; mass=missing, noprior=false)
    if noprior
        @assert !ismissing(attribute)
    else
        attribute ~ arraydist([Uniform(100, 300)])
    end
    mass, = attribute
    if eltype(scenario.scenes) <: Missing
        @assert "FallScenario doesn't support generation."
    else
        scenario = newby(scenario) do entity, group
            entity.mass[1] = mass
            entity
        end
        Turing.acclogp!(_varinfo, logpdf(like, scenario, getforce))
    end
end

# NOTE This function is not used anywhere
function getforce_fall(entity, state; kwargs...)
    f = zeros(si.velocity |> size)
    return f .+ [0, -9.8] .* entity.mass'
end

function loadpreset_fall()
    return (
        ScenarioModel=FallScenario,
        latentname=["mass"],
        ealg = ImportanceSampling(nsamples=200, rsamples=3),
        malg = BSPForce(
            grammar=G_BSP, opt=CrossEntropy(400, 2, 10, 300, 100.0), beta=1e1
        ),
        elike = Likelihood(nahead=2, nlevel=0.1),
        mlike = Likelihood(nahead=2, nlevel=0.1),
    )
end
