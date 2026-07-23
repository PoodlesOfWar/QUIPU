"""The resonance gate wired into resuscitation.

map_resuscitation_quipu() stamps a revival weight across all 4096 torus nodes.
Wired in, the weight of an *occupied* cell is scaled by the resonance of the
concept there (0.4x for a lonely caller ... 1.0x on the shared wavelength), so
recovery revives high-resonance shared concepts preferentially. This test runs
the same mesh with the gate off then on and checks that a high-resonance
concept's cell is boosted more (relative to its own geometric baseline) than a
low-resonance concept's cell — isolating resonance from the per-node phase.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest


@pytest.fixture
def placed_mesh(tmp_path, monkeypatch):
    db_file = tmp_path / "resus.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)
    mesh_slm.state_summary()

    cn = sqlite3.connect(str(db_file))
    cn.row_factory = sqlite3.Row
    # cells: cores along row 10, central at (1,1), lonely at (2,2), leaf at (3,3)
    placements = {
        "c0": (10, 0), "c1": (10, 1), "c2": (10, 2),
        "c3": (10, 3), "c4": (10, 4), "c5": (10, 5),
        "central": (1, 1), "lonely": (2, 2), "leaf": (3, 3),
    }
    for tok, (i, j) in placements.items():
        cn.execute("INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
                   "VALUES (?, ?, ?, 20, '2026-07-22T00:00:00+00:00', '2026-07-22T00:00:00+00:00')",
                   (tok, i, j))
    ids = {r["token"]: r["token_id"] for r in cn.execute("SELECT token, token_id FROM mesh_slm_vocab")}
    for tok in placements:                                  # all analytical (isolate centrality)
        cn.execute("INSERT INTO mesh_slm_embed(token_id, e_vision, e_touch, e_smell, e_body, "
                   "e_brain, e_perception, e_entirety) VALUES (?, 0, 0, 0, 0, 1.0, 0, 0)", (ids[tok],))

    def edge(a, b):
        cn.execute("INSERT OR IGNORE INTO mesh_slm_quipu(src, dst, weight, samples) "
                   "VALUES (?, ?, 1.0, 5)", (ids[a], ids[b]))
    core = ["c0", "c1", "c2", "c3", "c4", "c5"]
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            edge(core[i], core[j])
    for hub in ("c0", "c1", "c2"):
        edge("central", hub)                                # central: high centrality
    edge("lonely", "leaf")                                  # lonely: low centrality
    cn.commit()
    cn.close()
    return {"mesh_slm": mesh_slm, "placements": placements, "db": str(db_file)}


def _weight_at(db, i, j):
    cn = sqlite3.connect(db)
    cn.row_factory = sqlite3.Row
    try:
        row = cn.execute("SELECT resuscitation_weight AS w FROM mesh_slm_quipu_node "
                         "WHERE i=? AND j=?", (i, j)).fetchone()
        return float(row["w"]) if row else None
    finally:
        cn.close()


def test_resuscitation_is_resonance_gated(placed_mesh, monkeypatch):
    mesh_slm = placed_mesh["mesh_slm"]
    db = placed_mesh["db"]

    # Legacy geometric baseline (gate off).
    monkeypatch.setenv("QUIPU_RESUSCITATION_RESONANCE", "0")
    off = mesh_slm.map_resuscitation_quipu()
    assert off["resonance_gated"] is False
    base_central = _weight_at(db, 1, 1)
    base_lonely = _weight_at(db, 2, 2)

    # Gate on.
    monkeypatch.setenv("QUIPU_RESUSCITATION_RESONANCE", "1")
    on = mesh_slm.map_resuscitation_quipu()
    assert on["resonance_gated"] is True
    assert on["resonant_nodes"] == len(placed_mesh["placements"])
    gated_central = _weight_at(db, 1, 1)
    gated_lonely = _weight_at(db, 2, 2)

    # Same cell, same phase -> the only difference is the resonance gain. The
    # high-resonance concept must be boosted more (relative to its own baseline).
    ratio_central = gated_central / base_central
    ratio_lonely = gated_lonely / base_lonely
    assert ratio_central > ratio_lonely


def test_resuscitation_gate_opt_out_is_pure_geometric(placed_mesh, monkeypatch):
    mesh_slm = placed_mesh["mesh_slm"]
    monkeypatch.setenv("QUIPU_RESUSCITATION_RESONANCE", "off")
    summary = mesh_slm.map_resuscitation_quipu()
    assert summary["resonance_gated"] is False
    assert summary["resonant_nodes"] == 0
