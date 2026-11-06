"""Tests for quipu.observer_service — endpoints, source routing, and /slm inference."""

import json

import pytest

from quipu import observer_service, mesh_slm


def test_canonical_source_routing():
    assert observer_service._canonical_source("loadopoly-ocr") == "loadopoly-ocr"
    assert observer_service._canonical_source("loadopoly") == "loadopoly-ocr"
    assert observer_service._canonical_source("bakugo") == "bakugo"
    assert observer_service._canonical_source("cardcenter") == "bakugo"
    assert observer_service._canonical_source("supply-chain-brain") == "supply-chain-brain"
    assert observer_service._canonical_source("scb-brain") == "supply-chain-brain"
    assert observer_service._canonical_source("scb") == "supply-chain-brain"
    assert observer_service._canonical_source("hubcore") == "hubcore"
    assert observer_service._canonical_source("hub") == "hubcore"
    assert observer_service._canonical_source("hub-floor") == "hub-floor"
    assert observer_service._canonical_source("floor") == "hub-floor"


def test_observe_and_guidance_for_scb():
    # Test observe from supply chain brain
    code, res = observer_service._observe({
        "source": "supply-chain-brain",
        "text": "hydraulic pump assembly lead time 14 days",
    })
    assert code == 200
    assert res["ok"] is True
    assert res["axis"] == "brain"

    # Test guidance for supply chain brain
    guidance = observer_service._guidance("supply-chain-brain", limit=20)
    assert guidance["ok"] is True
    assert guidance["source"] == "supply-chain-brain"
    assert "lexicon" in guidance
    assert "world_model" in guidance


def test_slm_inference_route():
    # Test classification
    code, res = observer_service._slm({
        "kind": "classify",
        "labels": ["fastener", "pump", "valve"],
        "prompt": "heavy hex bolt grade 8",
    })
    assert code == 200
    assert res["ok"] is True
    assert "result" in res

    # Test generation
    code, res = observer_service._slm({
        "prompt": "inventory cycle count",
        "max_new_tokens": 10,
    })
    assert code == 200
    assert res["ok"] is True
    assert "text" in res["result"]


def test_observe_and_guidance_oscillation_coupling():
    # Test observe with meta.oscillation
    osc_input = {
        "phi": 0.812345,
        "theta": 0.123456,
        "omega": 0.010472,
        "kappa": 0.450000,
        "coherence": 0.880000,
        "gradient": 0.150000,
        "emergence": 0.750000,
        "temperature": 0.350000,
    }
    code, res = observer_service._observe({
        "source": "supply-chain-brain",
        "text": "analog wave phase lock check with Floquet resonance",
        "meta": {
            "oscillation": osc_input,
        },
    })
    assert code == 200
    assert res["ok"] is True
    assert "oscillation" in res
    assert res["oscillation"]["phi"] == 0.812345

    # Test guidance reflects oscillation
    guidance = observer_service._guidance("supply-chain-brain", limit=10)
    assert guidance["ok"] is True
    assert "oscillation" in guidance
    assert guidance["oscillation"]["phi"] == 0.812345
    assert guidance["oscillation"]["coherence"] == 0.88
    assert guidance["oscillation"]["temperature_bias"] == 0.35


def test_observer_world_model_transition_route():
    # Test step transition
    code, res = observer_service._world_model_transition({
        "state": {
            "inventory": 200.0,
            "demand_rate": 40.0,
        },
        "action": {
            "order_quantity": 40.0,
            "demand_multiplier": 1.5,
        },
    })
    assert code == 200
    assert res["ok"] is True
    assert res["mode"] == "step"
    # Available: 200. Demand: 40 * 1.5 = 60. Fulfilled: 60. Next inv: 140.
    assert res["predicted_state"]["inventory"] == 140.0
    assert res["predicted_state"]["flux"] == 60.0

    # Test rollout simulation
    code, res = observer_service._world_model_transition({
        "state": {"inventory": 100.0, "demand_rate": 20.0},
        "actions": [
            {"order_quantity": 20.0},
            {"order_quantity": 20.0},
        ],
    })
    assert code == 200
    assert res["ok"] is True
    assert res["mode"] == "rollout"
    assert len(res["trajectory"]) == 3




def test_observe_records_supernova_event_and_guidance_carries_the_gate(monkeypatch):
    from quipu import supernova_gate
    monkeypatch.setattr(supernova_gate, "_LAST_TS", 0.0)
    code, res = observer_service._observe({
        "source": "supply-chain-brain",
        "text": "theseuscape npc local expansion supernova decision",
        "meta": {"supernova": {"magnitude": 0.6, "direction": 1.2, "label": "npc-7 expands"}},
    })
    assert code == 200 and res["ok"] is True
    assert res["supernova"]["magnitude"] == 0.6
    assert res["supernova"]["source"] == "theseuscape"
    assert "supernova_gate" in res and 0.0 <= res["supernova_gate"]["gate"] <= 1.0
    guidance = observer_service._guidance("supply-chain-brain", limit=10)
    assert guidance["supernova_gate"]["events_in_window"] >= 1
    assert guidance["supernova_gate"]["recent_events"][-1]["label"] == "npc-7 expands"
    assert set(guidance["bond_shear"]) == {"yield", "shear_ratio", "litasb_rupture_nn"}


def test_observe_rejects_a_malformed_supernova_event():
    code, res = observer_service._observe({
        "source": "supply-chain-brain",
        "text": "malformed supernova payload",
        "meta": {"supernova": {"magnitude": "enormous"}},
    })
    assert code == 400 and res["ok"] is False
    assert "supernova" in res["error"]


def test_bond_shear_payload_reflects_the_published_chain():
    import sqlite3
    from quipu import brain_kv, local_store
    from quipu.qpsi import bond_shear
    cn = sqlite3.connect(local_store.db_path())
    rec = bond_shear.publish(cn, now=5.0)
    cn.commit()
    cn.close()
    payload = observer_service._bond_shear_payload()
    assert payload["yield"] == pytest.approx(rec["yield"])
    assert payload["shear_ratio"] == pytest.approx(rec["shear_ratio"])
    assert payload["litasb_rupture_nn"] == pytest.approx(rec["rupture"]["litasb_nn"])
    assert brain_kv.kv_get_json("entirety:bond_shear:latest", {})["yield"] == rec["yield"]
