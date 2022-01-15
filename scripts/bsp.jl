using Random, Logging, Turing, BayesianSymbolic
using ExprOptimization.ExprRules

includef(args...) = isdefined(Main, :Revise) ? includet(args...) : include(args...)

macro includeall(ullman=false)
    scenario_ullman = quote
        includef(srcdir("scenarios", "ullman.jl"))
    end
    hacks = quote
        includef(scriptsdir("ullman_hacks.jl"))
    end
    scenarios = quote
        includef(srcdir("scenarios", "magnet.jl"))
        includef(srcdir("scenarios", "nbody.jl"))
        includef(srcdir("scenarios", "bounce.jl"))
        includef(srcdir("scenarios", "mat.jl"))
        includef(srcdir("scenarios", "spring.jl"))
        includef(srcdir("scenarios", "fall.jl"))
        $(ullman ? scenario_ullman : :())
    end
    quote
        includef(srcdir("utility.jl"))
        includef(srcdir("app_inf.jl"))
        includef(srcdir("sym_reg.jl"))
        includef(srcdir("network.jl"))
        includef(srcdir("exp_max.jl"))
        includef(srcdir("analyse.jl"))
        includef(srcdir("dataset.jl"))
        # Suppress warnings of using _varinfo
        with_logger(SimpleLogger(stderr, Logging.Error)) do
            $scenarios
        end
        $(ullman ? hacks : :())
    end
end