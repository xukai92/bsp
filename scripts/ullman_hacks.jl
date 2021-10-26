babs(ex) = abs.(ex)

# NOTE W/ the knowledge of firction and bounce + a modified grammar for ULLMAN
# 1. Add `ConstVec`
# 2. Remove `Bool`
# 3. Remove `Unitless`
# 4. Replace `Kg` with `C`
# 5. Remove `lj`
# NOTE xor(q1, q2) can be learned by abs(qi + qj) / 2 where qi, qj \in {-1, 0, 1}
G_BSP_ULLMAN = @grammar begin
    # Constants
    Const = c1 | c2 | c3
    ConstVec = [0, 1] | [1, 0]
    # Coulomb
    C = qi | qj | badd(qi, qj) | bsub(qi, qj)
    CSq = bmul(qi, qj) | bpow2(C)
    # Basic vectors
    MeterVec = bsub(pi, pj)
    MeterSecVec = vi | vj | bsub(vi, vj)
    # Meter
    Meter = bnorm(MeterVec) | bproject(MeterVec, UnitlessVec)
    MeterSq = bpow2(Meter)
    # Meter per second
    MeterSec = bnorm(MeterSecVec) | bproject(MeterSecVec, UnitlessVec)
    MeterSecSq = bpow2(MeterSec)
    # Translation-invariant vectors
    TransInvVec = MeterVec | MeterSecVec
    # Unitless vectors
    UnitlessVec = bnormalize(TransInvVec) | bdiv(MeterVec, Meter) | bdiv(MeterSecVec, MeterSec) | ConstVec
    # BasicCoeff
    BasicCoeff = C | CSq | Meter | MeterSq | MeterSec | MeterSecSq
    BasicCoeff = bdiv(CSq, C) #| bsub(Meter, Meter) | badd(Meter, Meter) | bsub(MeterSq, MeterSq) | badd(MeterSq, MeterSq)
    # Coefficient
    Coeff = BasicCoeff | bmul(BasicCoeff, BasicCoeff) | bdiv(BasicCoeff, BasicCoeff) | babs(BasicCoeff)
    # Force; defined as magnitude * [condition *] direction
    BasicForce = Const * bmul(Coeff, UnitlessVec) # uncondiitonal force
    Force = BasicForce | badd(Force, BasicForce)
end

function gen_getforce_with_constant(grammar, ex::Expr; constant_scale=1, mass_scale=1, external=nothing)
    symtab = SymbolTable(grammar)
    function getforce_pair(etti::Entity, si::State, ettj::Entity, sj::State, constants)
        cs = constants * constant_scale
        f_shaped = zeros(size(si)...)
        nt = (
            qi = etti.charge,
            qj = ettj.charge,
            pi = si.position,
            pj = sj.position,
            vi = si.velocity,
            vj = sj.velocity,
            c1 = cs[1],
            c2 = cs[2],
            c3 = cs[3],
        )
        return f_shaped .+ apply(symtab, ex, nt)
    end
    if !isnothing(external)
        return (e, s, constants) -> getforce_pairwise(e, s, (ei, si, ej, sj) -> getforce_pair(ei, si, ej, sj, constants)) .+ external(e, s)
    else
        return (e, s, constants) -> getforce_pairwise(e, s, (ei, si, ej, sj) -> getforce_pair(ei, si, ej, sj, constants))
    end
end
