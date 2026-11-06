# vnn_bathymetry_model.jl
#
# QUIPU Viscous Neural Network (VNN) — Bathymetry, Vertical-Dynamics (v0.33),
# Exner Morphodynamics & Temporal-Spatial Reconstruction (Julia Reference Model).
#
# Mathematical Equations:
# 1. Bathymetric Depth & Slope:
#        H(x, y, t) = -B(x, y, t),    s(x, y) = ||∇H(x, y)||_2
# 2. Depth-Dependent Current Attenuation:
#        U(z) = U_0 * exp(-z / L)
# 3. Vertical-Motion Equation:
#        (m + m_a) dv/dt = (m - ρ_w V) g - (1/2) ρ_w C_D A v |v|,   dz/dt = v
# 4. Exner Sediment Continuity:
#        (1 - λ_p) ∂B/∂t = -∇ · q_s + S
# 5. Historical Bathymetry Reconstruction & Temporal Depression:
#        B_{1850}(x, y) = B_{modern}(x, y) - ∫_{1850}^{t_{modern}} (∂B/∂t) dt
#        D_T(x, y, t_0, t_1) = B(x, y, t_0) - B(x, y, t_1)
# 6. 6-Feature VNN Bathymetric Adapter:
#        (modern_depth_m, historical_depth_m, depth_uncertainty_m,
#         sediment_flux_x, sediment_flux_y, seabed_slope)

module QuipuVNNBathymetry

export BathymetricGrid2D, DescentParams, BathymetricFeatures6,
       interpolate_depth, seabed_slope, attenuate_current,
       vertical_acceleration, simulate_descent, exner_step!,
       reconstruct_historical_bathymetry, temporal_depression,
       extract_bathymetric_features, run_v033_synthetic_checks

struct BathymetricGrid2D
    x_coords::Vector{Float64}
    y_coords::Vector{Float64}
    elevation_b::Matrix{Float64}    # B(x, y, t) <= 0 for submerged seabed
    uncertainty_m::Matrix{Float64}
    timestamp::Float64
end

struct DescentParams
    mass_kg::Float64
    volume_m3::Float64
    drag_coeff::Float64
    area_m2::Float64
    added_mass_coeff::Float64
    rho_w::Float64
    g::Float64
end

function illustrative_v033_params()
    # Calibrated so a 100 m artificial seabed is reached in ~31.48 seconds
    return DescentParams(500.0, 0.15, 0.80, 0.8012, 0.50, 1025.0, 9.81)
end

struct BathymetricFeatures6
    modern_depth_m::Float64
    historical_depth_m::Float64
    depth_uncertainty_m::Float64
    sediment_flux_x::Float64
    sediment_flux_y::Float64
    seabed_slope::Float64
end

function bilinear_sample(grid::BathymetricGrid2D, field::Matrix{Float64}, x::Float64, y::Float64)
    xs = grid.x_coords
    ys = grid.y_coords
    xc = clamp(x, xs[1], xs[end])
    yc = clamp(y, ys[1], ys[end])
    ix = clamp(searchsortedlast(xs, xc), 1, length(xs) - 1)
    iy = clamp(searchsortedlast(ys, yc), 1, length(ys) - 1)
    tx = (xc - xs[ix]) / max(1e-12, xs[ix + 1] - xs[ix])
    ty = (yc - ys[iy]) / max(1e-12, ys[iy + 1] - ys[iy])
    return (1.0 - tx) * (1.0 - ty) * field[ix, iy] +
           tx * (1.0 - ty) * field[ix + 1, iy] +
           (1.0 - tx) * ty * field[ix, iy + 1] +
           tx * ty * field[ix + 1, iy + 1]
end

interpolate_elevation(grid::BathymetricGrid2D, x::Float64, y::Float64) =
    bilinear_sample(grid, grid.elevation_b, x, y)

interpolate_depth(grid::BathymetricGrid2D, x::Float64, y::Float64) =
    -interpolate_elevation(grid, x, y)

function seabed_slope(grid::BathymetricGrid2D, x::Float64, y::Float64)
    hx = max(1e-4, 0.25 * (grid.x_coords[2] - grid.x_coords[1]))
    hy = max(1e-4, 0.25 * (grid.y_coords[2] - grid.y_coords[1]))
    x_lo = max(grid.x_coords[1], x - hx)
    x_hi = min(grid.x_coords[end], x + hx)
    y_lo = max(grid.y_coords[1], y - hy)
    y_hi = min(grid.y_coords[end], y + hy)
    dh_dx = (interpolate_depth(grid, x_hi, y) - interpolate_depth(grid, x_lo, y)) / max(1e-12, x_hi - x_lo)
    dh_dy = (interpolate_depth(grid, x, y_hi) - interpolate_depth(grid, x, y_lo)) / max(1e-12, y_hi - y_lo)
    return hypot(dh_dx, dh_dy)
end

function attenuate_current(u0::Tuple{Float64, Float64}, z_m::Float64, L_m::Float64 = 40.0)
    fac = exp(-max(0.0, z_m) / max(1e-9, L_m))
    return (u0[1] * fac, u0[2] * fac)
end

function vertical_acceleration(v::Float64, p::DescentParams)
    m_a = p.added_mass_coeff * p.rho_w * p.volume_m3
    f_net = (p.mass_kg - p.rho_w * p.volume_m3) * p.g
    f_drag = 0.5 * p.rho_w * p.drag_coeff * p.area_m2 * v * abs(v)
    return (f_net - f_drag) / (p.mass_kg + m_a)
end

function simulate_descent(
    p::DescentParams,
    grid::BathymetricGrid2D;
    x0::Float64 = 500.0,
    y0::Float64 = 500.0,
    z0::Float64 = 0.0,
    v0::Float64 = 0.0,
    u_surface::Tuple{Float64, Float64} = (0.6, 0.2),
    L_m::Float64 = 40.0,
    dt::Float64 = 0.01,
    t_max::Float64 = 600.0,
)
    x, y, z, v, t = x0, y0, z0, v0, 0.0
    h_loc = interpolate_depth(grid, x, y)
    max_steps = ceil(Int, t_max / dt)
    for _ in 1:max_steps
        prev_x, prev_y, prev_z, prev_v, prev_t = x, y, z, v, t
        prev_clear = h_loc - z
        a1 = vertical_acceleration(v, p)
        v_mid = v + 0.5 * dt * a1
        z_mid = max(0.0, z + 0.5 * dt * v)
        a2 = vertical_acceleration(v_mid, p)
        v = v + dt * a2
        z = max(0.0, z + dt * v_mid)
        uh = attenuate_current(u_surface, z_mid, L_m)
        x += dt * uh[1]
        y += dt * uh[2]
        t += dt
        h_loc = interpolate_depth(grid, x, y)
        if z >= h_loc
            pen = z - h_loc
            alpha = clamp(prev_clear / max(1e-12, prev_clear + pen), 0.0, 1.0)
            t_hit = prev_t + alpha * dt
            x_hit = prev_x + alpha * (x - prev_x)
            y_hit = prev_y + alpha * (y - prev_y)
            return (contact = true, contact_time_s = t_hit, x = x_hit, y = y_hit, z = h_loc, v = v)
        end
    end
    return (contact = false, contact_time_s = NaN, x = x, y = y, z = z, v = v)
end

function exner_step!(
    grid::BathymetricGrid2D,
    q_sx::Matrix{Float64},
    q_sy::Matrix{Float64},
    source_sink::Matrix{Float64};
    lambda_p::Float64 = 0.40,
    dt_years::Float64 = 1.0,
)
    nx, ny = size(grid.elevation_b)
    dx = grid.x_coords[2] - grid.x_coords[1]
    dy = grid.y_coords[2] - grid.y_coords[1]
    max_delta = 0.0
    for i in 1:nx, j in 1:ny
        ip = min(nx, i + 1); im = max(1, i - 1)
        jp = min(ny, j + 1); jm = max(1, j - 1)
        dqx = (q_sx[ip, j] - q_sx[im, j]) / ((ip - im) * dx)
        dqy = (q_sy[i, jp] - q_sy[i, jm]) / ((jp - jm) * dy)
        db_dt = (-(dqx + dqy) + source_sink[i, j]) / (1.0 - lambda_p)
        delta = dt_years * db_dt
        grid.elevation_b[i, j] += delta
        max_delta = max(max_delta, abs(delta))
    end
    return max_delta
end

function reconstruct_historical_bathymetry(
    modern::BathymetricGrid2D;
    t_historical::Float64 = 1850.0,
    sedimentation_rate_m_yr::Float64 = 0.004,
    lambda_p::Float64 = 0.40,
)
    span = modern.timestamp - t_historical
    db = (sedimentation_rate_m_yr / (1.0 - lambda_p)) * span
    b_hist = modern.elevation_b .- db
    u_hist = sqrt.(modern.uncertainty_m .^ 2 .+ (0.15 * db)^2)
    return BathymetricGrid2D(copy(modern.x_coords), copy(modern.y_coords), b_hist, u_hist, t_historical)
end

temporal_depression(grid_t0::BathymetricGrid2D, grid_t1::BathymetricGrid2D, x::Float64, y::Float64) =
    interpolate_elevation(grid_t0, x, y) - interpolate_elevation(grid_t1, x, y)

function extract_bathymetric_features(
    x::Float64,
    y::Float64,
    modern::BathymetricGrid2D,
    historical::BathymetricGrid2D;
    flux_x::Float64 = 0.0,
    flux_y::Float64 = 0.0,
)
    h_mod = interpolate_depth(modern, x, y)
    h_hist = interpolate_depth(historical, x, y)
    u_mod = bilinear_sample(modern, modern.uncertainty_m, x, y)
    u_hist = bilinear_sample(historical, historical.uncertainty_m, x, y)
    s = seabed_slope(modern, x, y)
    return BathymetricFeatures6(h_mod, h_hist, max(u_mod, u_hist), flux_x, flux_y, s)
end

function run_v033_synthetic_checks()
    xs = collect(range(0.0, 1000.0, length = 21))
    ys = collect(range(0.0, 1000.0, length = 21))
    b = fill(-100.0, 21, 21)
    u = fill(0.5, 21, 21)
    grid = BathymetricGrid2D(xs, ys, b, u, 2026.0)
    p = illustrative_v033_params()
    out = simulate_descent(p, grid)
    qx = zeros(21, 21); qy = zeros(21, 21); s = zeros(21, 21)
    max_d = exner_step!(grid, qx, qy, s)
    return (
        depth_ok = isapprox(interpolate_depth(grid, 500.0, 500.0), 100.0; atol = 1e-9),
        atten_ok = isapprox(attenuate_current((1.0, 0.0), 40.0, 40.0)[1], exp(-1.0); atol = 1e-9),
        accel_ok = vertical_acceleration(0.0, p) > 0.0,
        contact_ok = out.contact && abs(out.contact_time_s - 31.48) <= 0.15,
        exner_zero_ok = max_d <= 1e-12,
        contact_time_s = out.contact_time_s,
    )
end

end # module QuipuVNNBathymetry
