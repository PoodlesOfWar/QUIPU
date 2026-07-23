"""Tests for doc_annealing.py — QUIPU Entirety living-map regeneration."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def isolated_anneal(tmp_path, monkeypatch):
    """Isolated mesh DB + freshly reloaded annealer bound to it."""
    db_file = tmp_path / "brain.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))

    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.brain_kv as bkv
    importlib.reload(bkv)
    import src.quipu.doc_annealing as da
    importlib.reload(da)
    return da


def _collected(*, mesh_density=0.18, certainty=0.28, nodal=0.21,
               version="0.26.0", root="hideout-mesh", roots=2):
    """A representative injected state dict, shaped like system_entirety_state()."""
    return {
        "version": version,
        "annealed_at": "2026-07-23T00:00:00+00:00",
        "entirety": {
            "axes": {"vision": 0.10, "touch": 0.0, "smell": 1.0,
                     "body": 0.72, "brain": 0.12, "perception": 0.0},
            "observer": 0.42,
            "magnitude": 1.3071,
            "material_bifurcation": {
                "physical_realization": 0.0,
                "nodal_bifurcation": nodal,
                "eligible": False,
                "mesh": {"mesh_density": mesh_density, "topology": "mesh"},
                "bridge_mesh_density": {
                    "primary_root": root, "total_roots": roots,
                    "mesh_density": 0.657, "tunnel_saturation": 1.0,
                    "anchor_strength": 0.376,
                },
            },
            "ueqgm_runtime": {
                "certainty": certainty, "symbiotic_gain": 0.54,
                "expansion_pressure": 0.51, "mesh_alignment": 0.184,
            },
            "transaction": {"drive": 0.31, "kind": "observer_only"},
        },
        "slm": {
            "vocab_size": 10, "vocab_capacity": 4096, "vocab_fill_pct": 0.24,
            "quipu_edges": 5, "rounds": 1, "last_loss": 0.1, "mesh_field_8d": 0.5,
            "last_stp_embed_gap": None, "last_stp_torus_gap": None,
        },
        "stp": {
            "insufficient_data": True, "window": 20, "loss_slope": None,
            "loss_plateaued": False, "stp_embed_slope": None,
            "stp_torus_slope": None, "p1_embed": False, "p1_torus": False,
        },
    }


def test_fingerprint_stable_and_sensitive(isolated_anneal):
    da = isolated_anneal
    base = da._fingerprint(_collected())
    # Identical structural state → identical fingerprint.
    assert da._fingerprint(_collected()) == base
    # Sub-0.01 jitter (rounds to the same 2 dp) → unchanged.
    assert da._fingerprint(_collected(mesh_density=0.184)) == base
    # A real structural move → different fingerprint.
    assert da._fingerprint(_collected(mesh_density=0.25)) != base
    assert da._fingerprint(_collected(root="other-root")) != base
    assert da._fingerprint(_collected(certainty=0.50)) != base


def test_structural_change_review_detects(isolated_anneal):
    da = isolated_anneal
    c = _collected()
    fp = da._fingerprint(c)
    assert da.structural_change_review(c, prev_hash=fp)["changed"] is False
    assert da.structural_change_review(c, prev_hash="deadbeef")["changed"] is True


def test_render_contains_sections(isolated_anneal):
    da = isolated_anneal
    text = da.render_system_map(_collected(), map_version=7, fingerprint="abc123")
    assert "# QUIPU Entirety — Living System Map" in text
    assert "Map revision**: v7" in text
    assert "MESH-SLM predictor" in text
    assert "score(t) = 0.55" in text            # predictor formula present
    assert "STP geodesic diagnostic" in text
    assert "hideout-mesh" in text               # live bridge root rendered
    assert "System Entirety — Live State" in text


def test_anneal_writes_increments_and_is_idempotent(isolated_anneal, tmp_path):
    da = isolated_anneal
    out = tmp_path / "map.md"
    c = _collected()

    first = da.anneal_docs(collected=c, out_path=str(out))
    assert first["status"] == "annealed"
    assert first["map_version"] == 1
    assert out.exists()

    # Unchanged structural state → no rewrite, version held.
    second = da.anneal_docs(collected=c, out_path=str(out))
    assert second["status"] == "unchanged"
    assert second["map_version"] == 1

    # force rewrites and bumps the map version.
    forced = da.anneal_docs(collected=c, out_path=str(out), force=True)
    assert forced["status"] == "annealed"
    assert forced["map_version"] == 2


def test_anneal_detects_structural_progression(isolated_anneal, tmp_path):
    da = isolated_anneal
    out = tmp_path / "map.md"
    da.anneal_docs(collected=_collected(mesh_density=0.18), out_path=str(out))
    # System progresses (mesh densifies) → new anneal fires.
    progressed = da.anneal_docs(collected=_collected(mesh_density=0.42), out_path=str(out))
    assert progressed["status"] == "annealed"
    assert progressed["map_version"] == 2
