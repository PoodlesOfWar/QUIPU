"""
    ViscousMCDistillationModel

Julia reference port of QUIPU's three extended mathematical subsystems:
1. Multidimensional incompressible Navier-Stokes (`∂u/∂t + (u·∇)u = -(1/ρ)∇p + ν∇²u + f`, `∇·u = 0`)
   and the Viscous Neural Network (VNN) nonlinear semantic-graph evolution
   (`dZ/dt = -ν_s L_G Z - κ_s N_G(Z) + F_sem(Z, t)`), connected via a typed
   coupling interface (`src/quipu/qpsi/viscous_interaction.py`).
2. Monte Carlo decision pruning (`J(a) = E[C(a)] + λ CVaR_β(C(a)) + ζ U(a)`)
   followed by weighted and greedy-Dijkstra next-path selection and centroidal
   deviation verification (`src/quipu/qpsi/mc_decision_pruning.py`).
3. QUIPU-native 5-term semantic distillation (`L = α L_CE + β T² L_KL + γ L_consistency
   + η L_complexity + ξ L_abstention`) with rADAM plasticity, Weyl compression,
   and shadow fallback/abstention gates (`src/quipu/qpsi/semantic_distillation.py`).
"""
module ViscousMCDistillationModel

export
    NavierStokes2D,
    taylor_green_2d,
    divergence_2d,
    project_incompressible_2d!,
    ns_step_2d!,
    VNNGraph,
    vnn_step!,
    compute_var_cvar,
    mc_edge_objective,
    mc_greedy_dijkstra,
    centroidal_deviation_7d,
    distillation_loss_5term,
    selftest

# ---------------------------------------------------------------------------
# 1. Multidimensional Navier-Stokes (Periodic 2D Grid) & VNN Graph Evolution
# ---------------------------------------------------------------------------

mutable struct NavierStokes2D
    nx::Int
    ny::Int
    lx::Float64
    ly::Float64
    rho::Float64
    nu::Float64
    dt::Float64
    u::Matrix{Float64}
    v::Matrix{Float64}
    p::Matrix{Float64}
end

function taylor_green_2d(n::Int = 16; rho::Float64 = 1.0, nu::Float64 = 0.02, dt::Float64 = 0.01, amp::Float64 = 0.5)
    u = zeros(Float64, n, n)
    v = zeros(Float64, n, n)
    p = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        x = 2π * (i - 1) / n
        y = 2π * (j - 1) / n
        u[i, j] =  amp * sin(x) * cos(y)
        v[i, j] = -amp * cos(x) * sin(y)
        p[i, j] = -0.25 * amp^2 * (cos(2x) + cos(2y))
    end
    return NavierStokes2D(n, n, 1.0, 1.0, rho, nu, dt, u, v, p)
end

@inline _idx(i::Int, n::Int) = mod1(i, n)

function divergence_2d(u::Matrix{Float64}, v::Matrix{Float64}, dx::Float64, dy::Float64)
    nx, ny = size(u)
    div = zeros(Float64, nx, ny)
    for i in 1:nx, j in 1:ny
        dudx = (u[_idx(i + 1, nx), j] - u[_idx(i - 1, nx), j]) / (2.0 * dx)
        dvdy = (v[i, _idx(j + 1, ny)] - v[i, _idx(j - 1, ny)]) / (2.0 * dy)
        div[i, j] = dudx + dvdy
    end
    return div
end

function project_incompressible_2d!(ns::NavierStokes2D; jacobi_iters::Int = 160)
    nx, ny = ns.nx, ns.ny
    dx, dy = ns.lx / nx, ns.ly / ny
    div = divergence_2d(ns.u, ns.v, dx, dy)
    rhs = (ns.rho / max(ns.dt, 1e-12)) .* div
    p = copy(ns.p)
    p_new = similar(p)
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)
    for _ in 1:jacobi_iters
        for i in 1:nx, j in 1:ny
            px = (p[_idx(i + 1, nx), j] + p[_idx(i - 1, nx), j]) / dx2
            py = (p[i, _idx(j + 1, ny)] + p[i, _idx(j - 1, ny)]) / dy2
            p_new[i, j] = (px + py - rhs[i, j]) / denom
        end
        mean_p = sum(p_new) / (nx * ny)
        p .= p_new .- mean_p
    end
    ns.p .= p
    for i in 1:nx, j in 1:ny
        dpdx = (p[_idx(i + 1, nx), j] - p[_idx(i - 1, nx), j]) / (2.0 * dx)
        dpdy = (p[i, _idx(j + 1, ny)] - p[i, _idx(j - 1, ny)]) / (2.0 * dy)
        ns.u[i, j] -= (ns.dt / ns.rho) * dpdx
        ns.v[i, j] -= (ns.dt / ns.rho) * dpdy
    end
    return ns
end

function ns_step_2d!(ns::NavierStokes2D)
    nx, ny = ns.nx, ns.ny
    dx, dy = ns.lx / nx, ns.ly / ny
    u0, v0 = copy(ns.u), copy(ns.v)
    for i in 1:nx, j in 1:ny
        ip, im = _idx(i + 1, nx), _idx(i - 1, nx)
        jp, jm = _idx(j + 1, ny), _idx(j - 1, ny)
        dudx = (u0[ip, j] - u0[im, j]) / (2.0 * dx)
        dudy = (u0[i, jp] - u0[i, jm]) / (2.0 * dy)
        dvdx = (v0[ip, j] - v0[im, j]) / (2.0 * dx)
        dvdy = (v0[i, jp] - v0[i, jm]) / (2.0 * dy)
        lap_u = (u0[ip, j] - 2.0 * u0[i, j] + u0[im, j]) / (dx * dx) +
                (u0[i, jp] - 2.0 * u0[i, j] + u0[i, jm]) / (dy * dy)
        lap_v = (v0[ip, j] - 2.0 * v0[i, j] + v0[im, j]) / (dx * dx) +
                (v0[i, jp] - 2.0 * v0[i, j] + v0[i, jm]) / (dy * dy)
        conv_u = u0[i, j] * dudx + v0[i, j] * dudy
        conv_v = u0[i, j] * dvdx + v0[i, j] * dvdy
        ns.u[i, j] = u0[i, j] + ns.dt * (-conv_u + ns.nu * lap_u)
        ns.v[i, j] = v0[i, j] + ns.dt * (-conv_v + ns.nu * lap_v)
    end
    project_incompressible_2d!(ns)
    return ns
end

mutable struct VNNGraph
    Z::Matrix{Float64}                   # (n_nodes, 7)
    edges::Vector{Tuple{Int,Int,Float64}}
    nu_s::Float64
    kappa_s::Float64
    dt::Float64
end

function vnn_step!(g::VNNGraph, target7::Vector{Float64}; forcing_gain::Float64 = 0.05)
    n = size(g.Z, 1)
    lap = zeros(Float64, n, 7)
    conv = zeros(Float64, n, 7)
    sym_deg = ones(Float64, n)
    out_deg = ones(Float64, n)
    for (u, v, w) in g.edges
        sym_deg[u] += 0.5 * w
        sym_deg[v] += 0.5 * w
        out_deg[u] += w
    end
    for (u, v, w) in g.edges
        duv = @views g.Z[v, :] .- g.Z[u, :]
        lap[u, :] .+= (0.5 * w / sym_deg[u]) .* (-duv)
        lap[v, :] .+= (0.5 * w / sym_deg[v]) .* duv
        nd = sqrt(sum(x * x for x in duv))
        if nd > 1e-12
            conv[u, :] .+= (w / out_deg[u]) * tanh(nd) .* duv
        end
    end
    for i in 1:n, d in 1:7
        f_id = forcing_gain * (target7[d] - g.Z[i, d])
        g.Z[i, d] = clamp(g.Z[i, d] + g.dt * (-g.nu_s * lap[i, d] - g.kappa_s * conv[i, d] + f_id), -2.0, 2.0)
    end
    return g.Z
end

# ---------------------------------------------------------------------------
# 2. Monte Carlo CVaR Decision Pruning & Greedy-Dijkstra
# ---------------------------------------------------------------------------

function compute_var_cvar(costs::AbstractVector{<:Real}, beta::Float64 = 0.90)
    isempty(costs) && return (0.0, 0.0)
    s = sort(Float64.(costs))
    n = length(s)
    k = clamp(ceil(Int, beta * n), 1, n)
    var_b = s[k]
    tail = @view s[k:n]
    cvar_b = sum(tail) / length(tail)
    return (var_b, cvar_b)
end

function mc_edge_objective(costs::AbstractVector{<:Real};
    beta::Float64 = 0.90, lam::Float64 = 0.35, zeta::Float64 = 0.25,
    eta_cc::Float64 = 0.02, samples::Int = 1, z_ci::Float64 = 1.96)
    m = length(costs)
    exp_c = sum(costs) / m
    std_c = m > 1 ? sqrt(sum((c - exp_c)^2 for c in costs) / m) : 0.0
    var_b, cvar_b = compute_var_cvar(costs, beta)
    uncert = z_ci * (std_c / sqrt(m)) + 0.25 / sqrt(max(0, samples) + 1.0)
    j = max(1e-6, exp_c + lam * cvar_b + zeta * uncert + eta_cc)
    return (expected = exp_c, var = var_b, cvar = cvar_b, uncertainty = uncert, j = j, ucb = exp_c + uncert)
end

function centroidal_deviation_7d(Z_path::Matrix{Float64}; p_bar::Float64 = 0.5)
    n = size(Z_path, 1)
    n == 0 && return (mean = 0.0, median = 0.0, mode = 0.0, linf = 0.0, inf_minus_1_lt_4p = true)
    centroid = [sum(Z_path[i, d] for i in 1:n) / n for d in 1:7]
    devs = Float64[]
    linf = 0.0
    for i in 1:n
        d2 = sqrt(sum((Z_path[i, d] - centroid[d])^2 for d in 1:7))
        dinf = maximum(abs(Z_path[i, d] - centroid[d]) for d in 1:7)
        push!(devs, d2)
        linf = max(linf, dinf)
    end
    s = sort(devs)
    med = isodd(n) ? s[(n + 1) ÷ 2] : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1])
    mean_d = sum(devs) / n
    bins = Dict{Int,Int}()
    for d in devs
        b = round(Int, d / 0.05)
        bins[b] = get(bins, b, 0) + 1
    end
    best_b = minimum(keys(bins))
    for (b, c) in bins
        if c > bins[best_b] || (c == bins[best_b] && b < best_b)
            best_b = b
        end
    end
    return (mean = mean_d, median = med, mode = best_b * 0.05, linf = linf,
            inf_minus_1_lt_4p = (linf - 1.0) < 4.0 * p_bar)
end

function mc_greedy_dijkstra(n_nodes::Int, edges::Vector{Tuple{Int,Int,Float64}}, src::Int, dst::Int)
    dist = fill(Inf, n_nodes)
    prev = zeros(Int, n_nodes)
    visited = falses(n_nodes)
    dist[src] = 0.0
    for _ in 1:n_nodes
        u = 0
        best = Inf
        for i in 1:n_nodes
            if !visited[i] && dist[i] < best
                best = dist[i]
                u = i
            end
        end
        (u == 0 || u == dst) && break
        visited[u] = true
        for (s, d, w) in edges
            if s == u && w > 0.0
                nd = dist[u] + w
                if nd < dist[d]
                    dist[d] = nd
                    prev[d] = u
                end
            end
        end
    end
    !isfinite(dist[dst]) && return (found = false, path = Int[], cost = Inf)
    path = Int[dst]
    while last(path) != src
        push!(path, prev[last(path)])
    end
    reverse!(path)
    return (found = true, path = path, cost = dist[dst])
end

# ---------------------------------------------------------------------------
# 3. QUIPU-Native 5-Term Semantic Distillation
# ---------------------------------------------------------------------------

function _softmax(logits::AbstractVector{<:Real}, T::Float64 = 1.0)
    scaled = Float64.(logits) ./ max(T, 1e-8)
    m = maximum(scaled)
    exps = exp.(scaled .- m)
    return exps ./ sum(exps)
end

function _kl(p::AbstractVector{Float64}, q::AbstractVector{Float64})
    s = 0.0
    for (pi, qi) in zip(p, q)
        if pi > 0.0
            s += pi * log(max(1e-12, pi) / max(1e-12, qi))
        end
    end
    return max(0.0, s)
end

function distillation_loss_5term(
    teacher_logits::AbstractVector{<:Real},
    student_logits::AbstractVector{<:Real};
    perturbed_logits::Union{Nothing,AbstractVector{<:Real}} = nothing,
    student_weights::AbstractVector{<:Real} = Float64[],
    active_ratio::Float64 = 0.25,
    T::Float64 = 2.0,
    alpha::Float64 = 0.25, beta::Float64 = 0.35, gamma::Float64 = 0.15,
    eta::Float64 = 0.10, xi::Float64 = 0.15,
    cost_wrong::Float64 = 0.65, cost_missed::Float64 = 0.35,
)
    p_t1 = _softmax(teacher_logits, 1.0)
    p_s1 = _softmax(student_logits, 1.0)
    l_ce = -sum(y * log(max(1e-12, ps)) for (y, ps) in zip(p_t1, p_s1))

    p_tT = _softmax(teacher_logits, T)
    p_sT = _softmax(student_logits, T)
    t2_kl = (T * T) * _kl(p_tT, p_sT)

    l_cons = if perturbed_logits === nothing
        0.0
    else
        p_sp = _softmax(perturbed_logits, 1.0)
        0.5 * (_kl(p_s1, p_sp) + _kl(p_sp, p_s1))
    end

    w_comp = isempty(student_weights) ? 0.0 :
        0.5 * (sum(abs2, student_weights) + sum(abs, student_weights)) / length(student_weights)
    l_comp = w_comp + clamp(active_ratio, 0.0, 1.0)

    n_c = max(2, length(p_s1))
    ent = -sum(p * log(max(1e-12, p)) for p in p_s1) / log(n_c)
    kl1 = _kl(p_t1, p_s1)
    abstain_p = clamp(0.6 * ent + 0.4 * tanh(kl1), 0.0, 1.0)
    p_star = cost_wrong / (cost_wrong + cost_missed)
    err_proxy = max(0.0, 1.0 - sum(p_t1 .* p_s1))
    l_abs = (1.0 - abstain_p) * err_proxy + abstain_p * p_star

    tot = alpha * l_ce + beta * t2_kl + gamma * l_cons + eta * l_comp + xi * l_abs
    return (total = tot, l_ce = l_ce, t2_l_kl = t2_kl, l_consistency = l_cons,
            l_complexity = l_comp, l_abstention = l_abs, abstention_prob = abstain_p, p_star = p_star)
end

function selftest()
    ns = taylor_green_2d(16)
    ns_step_2d!(ns)
    div = divergence_2d(ns.u, ns.v, ns.lx / ns.nx, ns.ly / ns.ny)
    @assert maximum(abs, div) < 1e-2

    g = VNNGraph(fill(0.2, 4, 7), [(1, 2, 0.8), (2, 3, 0.6), (3, 4, 0.9)], 0.08, 0.04, 0.1)
    vnn_step!(g, fill(0.5, 7))
    @assert all(isfinite, g.Z)

    mc = mc_edge_objective([0.2, 0.25, 0.3, 0.5, 0.8]; beta = 0.80)
    @assert mc.cvar >= mc.var >= mc.expected
    rt = mc_greedy_dijkstra(4, [(1, 2, mc.j), (2, 4, 0.3), (1, 3, 0.9), (3, 4, 0.9)], 1, 4)
    @assert rt.found && rt.path == [1, 2, 4]

    cd = centroidal_deviation_7d(g.Z)
    @assert cd.inf_minus_1_lt_4p

    dl = distillation_loss_5term([2.0, 0.5, -1.0], [1.8, 0.4, -0.9];
        perturbed_logits = [1.7, 0.45, -0.85], student_weights = fill(0.5, 7))
    @assert isfinite(dl.total) && dl.total > 0.0
    return true
end

end # module
