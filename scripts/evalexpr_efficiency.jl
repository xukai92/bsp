using InteractiveUtils, DrWatson
versioninfo(); @quickactivate

using Comonicon, Test, BenchmarkTools, ExprOptimization
using ExprOptimization.ProbabilisticExprRules: RuleNode, mindepth_map, ProbabilisticGrammar
using GeneralizedGenerated: mk_function
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
RuntimeGeneratedFunctions.init(@__MODULE__)

const G = @grammar begin
    Real = m1
    Real = m2
    Real = r²
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real
    Real = Real / Real
end

const PCFG = ProbabilisticGrammar(G)

# NOTE This version fails due to the world age problem.
# function gen_f_invoke(tree)
#     ex = get_executable(tree, G)
#     f = eval(:((m1, m2, r²) -> $ex))
#     return (m1, m2, r²) -> f(m1, m2, r²)
# end

function gen_f_invoke(tree)
    ex = get_executable(tree, G)
    f = eval(:((m1, m2, r²) -> $ex))
    return (m1, m2, r²) -> Base.invokelatest(f, m1, m2, r²)
end

function gen_f_mkfunc(tree)
    ex = get_executable(tree, G)
    return mk_function(:((m1, m2, r²) -> $ex))
end

function gen_f_rgfunc(tree)
    ex = get_executable(tree, G)
    return @RuntimeGeneratedFunction :((m1, m2, r²) -> $ex)
end

function gen_f_letblk(tree)
    ex = get_executable(tree, G)
    return function f(m1, m2, r²)
        return eval(
            quote
                let m1 = $m1, m2 = $m2, r² = $r²
                    $ex
                end
            end
        )
    end
end

"""
N: number of random tree draws
D: depth of trees to sample
"""
@main function main(N::Int, D::Int)
    t = Dict(:invoke => zeros(N),
             :mkfunc => zeros(N),
             :rgfunc => zeros(N),
             :letblk => zeros(N))
    
    Threads.@threads for i in 1:N
        m1, m2, r² = rand(3)
        tree = rand(RuleNode, PCFG, :Real, mindepth_map(G), D)
        f_invoke = gen_f_invoke(tree)
        f_mkfunc = gen_f_mkfunc(tree)
        f_rgfunc = gen_f_rgfunc(tree)
        f_letblk = gen_f_letblk(tree)
        @test f_invoke(m1, m2, r²) == f_mkfunc(m1, m2, r²)
        @test f_invoke(m1, m2, r²) == f_rgfunc(m1, m2, r²)
        @test f_invoke(m1, m2, r²) == f_letblk(m1, m2, r²)
        t[:invoke][i] = @belapsed $f_invoke($m1, $m2, $r²)
        t[:mkfunc][i] = @belapsed $f_mkfunc($m1, $m2, $r²)
        t[:rgfunc][i] = @belapsed $f_rgfunc($m1, $m2, $r²)
        t[:letblk][i] = @belapsed $f_letblk($m1, $m2, $r²)
    end
    
    t = (invoke=sum(t[:invoke]),
         mkfunc=sum(t[:mkfunc]),
         rgfunc=sum(t[:rgfunc]),
         letblk=sum(t[:letblk]))
    @info "Benchmark results for N=$N, D=$D" t...
end

"""
```
┌ Info: Benchmark results for N=10, D=10
│   invoke = 5.145409596755064e-7
│   mkfunc = 5.58e-9
│   rgfunc = 5.58e-9
└   letblk = 0.004392728
```
"""
