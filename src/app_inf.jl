using StatsFuns: logsumexp
using Distributions: Categorical

function get_samples(chain, ks::Vector{String})
    logweights = vec(collect(chain[:lp]))
    values = chain[ks].value.data[:,:,1] # 1 is the chain index; here a single chain is assumed
    return [WeightedSample(logweight=logweights[i], value=values[i,:]) for i in 1:length(logweights)]
end

function expect(f::Function, wxs::Vector{<:WeightedSample})
    n = length(wxs)
    # NOTE We normalise the weights (in a numerically safe manner) first so that logZ is 0
    lws = map(wx -> wx.logweight, wxs)
    lws = lws .- logsumexp(lws)
    return mapreduce(i -> exp(lws[i]) * f(wxs[i].value), +, 1:n)
end

function expect(f::Function, chain::Chains, ks::Vector{String})
    wxs = get_samples(chain, ks)
    return expect(f, wxs)
end

function rsample(wxs::Vector{<:WeightedSample}, n::Int)
    lws = map(wx -> wx.logweight, wxs)
    lws = lws .- logsumexp(lws)
    ws = exp.(lws)
    idcs = rand(Categorical(ws), n)
    # NOTE Returned samples are unnormlized.
    return wxs[idcs]
end

function rsample(chain::Chains, ks::Vector{String}, n::Int)
    wxs = get_samples(chain, ks)
    return n > 0 ? rsample(wxs, n) : wxs
end

function rsample(chain::Chains, n::Int)
    nattributes = size(chain, 2) - 1
    ks = ["attribute[$i]" for i in 1:nattributes]
    return rsample(chain, ks, n)
end
