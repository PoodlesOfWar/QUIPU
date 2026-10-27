"""Acceptance tests for QUIPU VNN Bathymetry (v0.33), Vertical-Dynamics,
Exner Morphodynamics, Temporal-Spatial Reconstruction, and Holon Fleet wiring.
"""
from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pytest

from quipu import geospatial_relation as gr
from quipu.qpsi import mc_decision_pruning as mcd
from quipu.qpsi import viscous_interaction as vi
from quipu.qpsi import vnn_bathymetry as vb


def _make_corpus_conn() -> sqlite3.Connection:
    cn = sqlite3.connect(":memory:")
    cn.row_factory = sqlite3.Row
    cn.executescript(
        """
        CREATE TABLE corpus_entity (
            entity_id TEXT,
            entity_type TEXT,
            label TEXT,
            props_json TEXT,
            first_seen TEXT,
            last_seen TEXT,
            samples INTEGER DEFAULT 1,
            PRIMARY KEY (entity_id, entity_type)
        );
        CREATE TABLE corpus_edge (
            src_id TEXT,
            src_type TEXT,
            dst_id TEXT,
            dst_type TEXT,
            rel TEXT,
            weight REAL,
            last_seen TEXT,
            samples INTEGER DEFAULT 1,
            PRIMARY KEY (src_id, src_type, dst_id, dst_type, rel)
        );
        CREATE TABLE mesh_slm_vocab (
            token_id INTEGER PRIMARY KEY,
            token TEXT UNIQUE,
            freq INTEGER DEFAULT 1,
            i INTEGER DEFAULT 0,
            j INTEGER DEFAULT 0,
            last_seen REAL DEFAULT 0.0
        );
        CREATE TABLE mesh_slm_embed (
            token_id INTEGER PRIMARY KEY,
            e_vision REAL, e_touch REAL, e_smell REAL, e_body REAL,
            e_brain REAL, e_perception REAL, e_entirety REAL
        );
        CREATE TABLE mesh_slm_quipu (
            src INTEGER,
            dst INTEGER,
            weight REAL,
            samples INTEGER DEFAULT 1,
            PRIMARY KEY (src, dst)
        );
        CREATE TABLE mesh_slm_quipu_node (
            node_id INTEGER PRIMARY KEY,
            resuscitation_weight REAL NOT NULL DEFAULT 0.8,
            directed_target INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    for tid in range(1, 5):
        cn.execute("INSERT INTO mesh_slm_vocab(token_id, token, freq) VALUES(?, ?, ?)", (tid, f"tok_{tid}", 10))
        cn.execute(
            "INSERT INTO mesh_slm_embed VALUES(?, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)",
            (tid,),
        )
    cn.execute("INSERT INTO mesh_slm_quipu(src, dst, weight, samples) VALUES(1, 2, 0.85, 8)")
    cn.execute("INSERT INTO mesh_slm_quipu(src, dst, weight, samples) VALUES(1, 3, 0.15, 1)")
    cn.execute("INSERT INTO mesh_slm_quipu(src, dst, weight, samples) VALUES(2, 4, 0.80, 6)")
    cn.execute("INSERT INTO mesh_slm_quipu(src, dst, weight, samples) VALUES(3, 4, 0.20, 1)")
    return cn


def test_v033_five_synthetic_checks_and_31_48s_descent():
    res = vb.run_v033_synthetic_checks()
    assert res["all_passed"] is True
    for k, status in res["checks"].items():
        assert status == "PASS", f"check {k} failed"
    # Verify the synthetic 100 m descent reaches seabed in ~31.48 seconds
    assert abs(res["synthetic_descent_time_s"] - 31.48) <= 0.05


def test_noaa_esri_ascii_and_gebco_xyz_grid_fusion():
    esri_text = """
    ncols 3
    nrows 3
    xllcorner 0.0
    yllcorner 0.0
    cellsize 50.0
    nodata_value -9999
    -102.0 -104.0 -106.0
    -100.0 -102.0 -104.0
    -98.0  -100.0 -102.0
    """
    noaa_grid = vb.load_esri_ascii_grid(esri_text, source="NOAA", default_uncertainty_m=1.0)
    assert noaa_grid.interpolate_depth(50.0, 50.0) == pytest.approx(102.0)

    csv_text = """x,y,elevation_b,uncertainty_m
    0.0,0.0,-99.0,3.0
    0.0,100.0,-103.0,3.0
    100.0,0.0,-103.0,3.0
    100.0,100.0,-107.0,3.0
    """
    gebco_grid = vb.load_xyz_csv_grid(csv_text, source="GEBCO")
    fused = vb.fuse_bathymetric_grids(noaa_grid, gebco_grid)
    # Inverse-variance fusion reduces uncertainty below both individual sources
    assert fused.interpolate_uncertainty(50.0, 50.0) < noaa_grid.interpolate_uncertainty(50.0, 50.0)
    assert fused.interpolate_uncertainty(50.0, 50.0) < gebco_grid.interpolate_uncertainty(50.0, 50.0)


def test_surface_drift_separates_ocean_atmosphere_wave_and_diffusion():
    pos0 = (100.0, 200.0)
    u_o = (0.5, 0.0)
    u_a = (10.0, 0.0)
    u_w = (0.0, 0.2)

    ocean_only = vb.step_surface_drift(
        pos0, u_o, u_a, u_w,
        vb.SurfaceDriftConfig(enable_ocean=True, enable_atmosphere=False, enable_waves=False, enable_diffusion=False, dt_s=100.0),
    )
    assert ocean_only["delta_ocean"] == pytest.approx((50.0, 0.0))
    assert ocean_only["delta_atmosphere"] == pytest.approx((0.0, 0.0))
    assert ocean_only["delta_wave"] == pytest.approx((0.0, 0.0))

    atmo_only = vb.step_surface_drift(
        pos0, u_o, u_a, u_w,
        vb.SurfaceDriftConfig(leeway_alpha=0.03, enable_ocean=False, enable_atmosphere=True, enable_waves=False, enable_diffusion=False, dt_s=100.0),
    )
    assert atmo_only["delta_atmosphere"] == pytest.approx((30.0, 0.0))
    assert atmo_only["delta_ocean"] == pytest.approx((0.0, 0.0))


def test_multi_regime_vessel_vs_person_and_extended_hull_contact():
    grid = vb.BathymetricGrid.sloped_shelf(base_depth_m=80.0, slope_x=0.04, slope_y=0.0)
    hull_params = vb.DescentObjectParams.helena_sloman_hull(phase_state="flooded_hull", pitch_rad=0.18)
    person_float = vb.DescentObjectParams.person_in_water(positively_buoyant=True)
    person_sub = person_float.with_phase_state("submerged")

    # Positively buoyant person stays at the surface (does not sink to 80 m seabed)
    res_person_float = vb.simulate_descent(person_float, grid, t_max_s=60.0, dt_s=0.05)
    assert res_person_float["seabed_contact"] is False
    assert res_person_float["landing_z_m"] == pytest.approx(0.0, abs=1e-4)

    # Negatively buoyant submerged person sinks
    assert vb.vertical_acceleration(0.0, person_sub) > 0.0

    # Extended hull detects bow/stern contact earlier than center-point contact on a sloped shelf
    res_point = vb.simulate_descent(hull_params, grid, dt_s=0.05, use_extended_hull=False)
    res_hull = vb.simulate_descent(hull_params, grid, dt_s=0.05, use_extended_hull=True)
    assert res_point["seabed_contact"] is True
    assert res_hull["seabed_contact"] is True
    assert res_hull["contact_details"]["mode"] == "extended_hull"
    assert res_hull["contact_time_s"] < res_point["contact_time_s"]


def test_exner_1850_reconstruction_temporal_depression_and_7_state_wreck_burial():
    modern = vb.BathymetricGrid.sloped_shelf(base_depth_m=95.0, timestamp=2026.0)
    hist_1850, summary = vb.reconstruct_historical_bathymetry(
        modern, t_historical=1850.0, source_sink=0.005
    )
    assert summary["span_years"] == pytest.approx(176.0)
    # Positive deposition since 1850 means 1850 seabed was deeper (lower B) than modern
    dep = vb.temporal_depression(hist_1850, modern, 500.0, 500.0)
    assert dep["temporal_depression_m"] < 0.0
    assert dep["interpretation"] == "multiscale_relational_elevation_difference"

    # Evaluate wreck morphodynamics & re-exposure without changing horizontal displacement
    w_buried = vb.evaluate_wreck_burial(
        seabed_elevation_b_m=-92.0,
        keel_elevation_b0_m=-96.5,
        hull_height_m=4.6,
        elapsed_years=176.0,
        bottom_current_speed_m_s=0.35,
        horizontal_displacement_m=0.0,
    )
    assert w_buried.burial_depth_m > 0.0
    assert w_buried.horizontal_displacement_m == 0.0
    assert "sediment_accumulation" in w_buried.active_lifecycle_states
    assert "structural_collapse" in w_buried.active_lifecycle_states

    w_reexposed = vb.evaluate_wreck_burial(
        seabed_elevation_b_m=-95.0,
        keel_elevation_b0_m=-96.5,
        hull_height_m=4.6,
        elapsed_years=176.0,
        prior_max_burial_m=2.5,
    )
    assert "re_exposure" in w_reexposed.active_lifecycle_states
    assert w_reexposed.dominant_state == "re_exposure"


def test_geodesic_graph_neg_log_prob_dijkstra_voronoi_and_pruned_mass_ledger():
    g = vb.GeodesicReconstructionGraph()
    g.add_node(
        vb.ReconstructionNode(
            node_id="n_start",
            entity_type="sighting_1850",
            latitude=40.50,
            longitude=-69.20,
            depth=0.0,
            timestamp=1850.0,
            physical_state={"phase": "floating"},
            evidence_source="logbook",
            uncertainty=500.0,
        )
    )
    g.add_node(
        vb.ReconstructionNode(
            node_id="n_mid_high",
            entity_type="drift_state",
            latitude=40.55,
            longitude=-69.15,
            depth=20.0,
            timestamp=1850.1,
            physical_state={"phase": "flooded_hull"},
            evidence_source="ocean_model",
            uncertainty=600.0,
        )
    )
    g.add_node(
        vb.ReconstructionNode(
            node_id="n_mid_low",
            entity_type="drift_state",
            latitude=40.60,
            longitude=-69.10,
            depth=20.0,
            timestamp=1850.1,
            physical_state={"phase": "flooded_hull"},
            evidence_source="ocean_model",
            uncertainty=900.0,
        )
    )
    g.add_node(
        vb.ReconstructionNode(
            node_id="n_end",
            entity_type="wreck_candidate",
            latitude=40.58,
            longitude=-69.12,
            depth=110.0,
            timestamp=1850.2,
            physical_state={"phase": "grounded"},
            evidence_source="sonar_prior",
            uncertainty=350.0,
        )
    )
    g.add_edge(vb.ReconstructionEdge.from_probability("n_start", "n_mid_high", "surface_drift", 0.80, 6500.0, 0.1))
    g.add_edge(vb.ReconstructionEdge.from_probability("n_start", "n_mid_low", "surface_drift", 0.04, 12000.0, 0.1))
    g.add_edge(vb.ReconstructionEdge.from_probability("n_mid_high", "n_end", "vertical_descent", 0.75, 4200.0, 0.1))
    g.add_edge(vb.ReconstructionEdge.from_probability("n_mid_low", "n_end", "vertical_descent", 0.50, 3100.0, 0.1))

    route = g.dijkstra_max_probability_path("n_start", "n_end", min_edge_probability=0.10)
    assert route["found"] is True
    assert route["path"] == ["n_start", "n_mid_high", "n_end"]
    assert route["path_probability"] == pytest.approx(0.80 * 0.75)
    ledger = route["pruned_mass_ledger"]
    assert ledger["pruned_branches"] == 1
    assert ledger["pruned_probability_mass"] == pytest.approx(0.04)

    vor = g.assign_voronoi_cells([(40.501, -69.201), (40.581, -69.121)])
    assert len(vor["n_start"]) == 1
    assert len(vor["n_end"]) == 1

    interp = g.fuzzy_interpolate_nodes("n_start", "n_end", alpha=0.5)
    assert interp.uncertainty > 0.5 * (500.0 + 350.0)


def test_shadow_mc_8_ensembles_helena_sloman_pompeii_and_exports():
    hs = vb.build_helena_sloman_1850_case()
    assert hs["case_id"] == "helena_sloman_1850"
    assert hs["hull_descent"]["seabed_contact"] is True
    assert hs["person_drift"]["seabed_contact"] is False
    ens = hs["shadow_mc"]["ensembles"]
    assert set(ens.keys()) == set(vb.SHADOW_ENSEMBLE_NAMES)

    geojson_str = vb.export_reconstruction_geojson(hs["shadow_mc"])
    parsed = json.loads(geojson_str)
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 8

    csv_str = vb.export_reconstruction_csv(hs["shadow_mc"])
    lines = [ln for ln in csv_str.strip().splitlines() if ln.strip()]
    assert len(lines) == 9  # header + 8 ensembles

    pompeii = vb.build_pompeii_79ce_case()
    assert pompeii["case_id"] == "pompeii_vesuvius_79ce"
    assert pompeii["temporal_depression"]["temporal_depression_m"] < 0.0


def test_geospatial_relation_planck_gate_and_typed_viscous_coupler_separation():
    cn = _make_corpus_conn()
    grid = vb.BathymetricGrid.sloped_shelf()
    hist, _ = vb.reconstruct_historical_bathymetry(grid, t_historical=1850.0)
    feat = vb.extract_bathymetric_features(500.0, 500.0, grid, hist, sediment_flux_x=0.02, sediment_flux_y=-0.01)

    p1 = gr.attach_bathymetry_features(
        cn, "sloman_site_1", feat, now="2026-10-27T00:00:00Z", cost_class="bathymetric_sea", flip_count=7
    )
    assert p1["planck_displaced"] is True
    assert p1["flip_count_at_displacement"] == 7
    s1 = cn.execute("SELECT samples FROM corpus_entity WHERE entity_id='sloman_site_1'").fetchone()[0]
    assert s1 == 1

    # Sub-Planck update refreshes last_seen without incrementing samples
    p2 = gr.attach_bathymetry_features(
        cn, "sloman_site_1", feat, now="2026-10-27T01:00:00Z", cost_class="bathymetric_sea", flip_count=9
    )
    s2 = cn.execute("SELECT samples FROM corpus_entity WHERE entity_id='sloman_site_1'").fetchone()[0]
    assert s2 == 1

    # Verify 7-D semantic embeddings in mesh_slm_embed remain untouched and CoupledStepReport stays separated
    ns_cfg = vi.NavierStokesConfig(dims=(8, 8), coriolis_f=1e-4)
    ns_state = vi.NavierStokesState.taylor_green(ns_cfg)
    vnn_state = vi.load_vnn_state_from_conn(cn)
    coupler = vi.TypedViscousCoupler(ns_cfg=ns_cfg)
    _, vnn_next, report = coupler.step(ns_state, vnn_state, bathymetry_features=feat.to_dict())
    assert vnn_next.Z.shape[1] == 7
    assert report.semantic.semantic_dims == 7
    assert report.bathymetric_feature_dims == 6
    assert report.bathymetry_features["modern_depth_m"] == pytest.approx(feat.modern_depth_m)

    # Verify MC pruning route records retained and pruned probability mass
    route = mcd.mc_dijkstra_route(cn, 1, 4)
    assert route["found"] is True
    assert 0.0 <= route["retained_probability_mass"] <= 1.0
    assert 0.0 <= route["pruned_probability_mass"] <= 1.0
    assert route["retained_probability_mass"] + route["pruned_probability_mass"] == pytest.approx(1.0, abs=1e-5)
