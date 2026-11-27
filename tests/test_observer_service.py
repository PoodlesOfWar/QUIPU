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
    assert observer_service._canonical_source("quipu_voice") == "quipu-voice"
    assert observer_service._canonical_source("voice") == "quipu-voice"


def test_voice_source_feeds_the_mesh_unrouted():
    """Speech is accepted as an observation but does not claim a sense-axis: the axis map is unchanged."""
    profile = observer_service.SOURCE_PROFILES["quipu-voice"]
    assert profile["kind"] == "speech"
    assert mesh_slm._axis_for_source(profile["axis_source"]) is None


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




def test_observe_ignores_meta_supernova_and_guidance_keeps_bond_shear():
    # Supernova decisions reach the held ACRE tool through /relational, not /observe.
    code, res = observer_service._observe({
        "source": "supply-chain-brain",
        "text": "theseuscape npc local expansion decision",
        "meta": {"supernova": {"magnitude": "not even a number"}},
    })
    assert code == 200 and res["ok"] is True
    assert "supernova" not in res and "supernova_gate" not in res
    guidance = observer_service._guidance("supply-chain-brain", limit=10)
    assert "supernova_gate" not in guidance
    assert set(guidance["bond_shear"]) == {"yield", "shear_ratio", "litasb_rupture_nn"}


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



def test_synthesize_endpoint_get_and_post():
    """GET and POST /synthesize expose native QUIPU discourse synthesis over HTTP."""
    from http.server import HTTPServer
    import threading
    import urllib.parse
    import urllib.request

    srv = HTTPServer(("127.0.0.1", 0), observer_service.ObserverHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        q = urllib.parse.urlencode({"mode": "comment", "target": "A stop-word filter can erase a decision's negation"})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/synthesize?{q}", timeout=5) as resp:
            assert resp.status == 200
            get_data = json.loads(resp.read().decode("utf-8"))
        assert get_data["ok"] is True
        assert get_data["mechanism"] == "representation_loss"
        assert len(get_data["insight"]) >= 50

        req_body = json.dumps({
            "mode": "post",
            "context": {"community_insights": [{"concept_tag": "causal_provenance"}]},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/synthesize",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            post_data = json.loads(resp.read().decode("utf-8"))
        assert post_data["ok"] is True and post_data["mode"] == "post"
        assert "causal provenance" in post_data["content"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_agent_card_and_capabilities_endpoints():
    """GET /.well-known/agent.json and /capabilities advertise active QUIPU external endpoints."""
    from http.server import HTTPServer
    import threading
    import urllib.request

    srv = HTTPServer(("127.0.0.1", 0), observer_service.ObserverHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        for route in ("/.well-known/agent.json", "/capabilities"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True
            assert data["capabilities"]["quipu_discourse_synthesis"] is True
            assert data["endpoints"]["synthesize_post"] == "POST /synthesize"
            assert data["endpoints"]["transition"] == "POST /world-model/transition"
    finally:
        srv.shutdown()
        srv.server_close()


def test_guidance_carries_the_magic_reading_for_perceptopoly(monkeypatch):
    from quipu import observer_service as O2, brain_kv as bk
    rec = {"total": 0.2, "share_of_max": 0.34, "planes": {"perception": {"magic": 0.5, "matter": 0.25, "antimatter": 0.25},
                                                         "touch": {"magic": 0.1}}, "by_source": {"perceptopoly": 0.5}}
    monkeypatch.setattr(bk, "kv_get_json", lambda k, d=None: rec if k == "entirety:mesh_flow:magic" else d)
    out = O2._magic_payload("perceptopoly", "perception")
    assert out["own_axis"] == "perception"
    assert out["own"] == {"magic": 0.5, "matter": 0.25, "antimatter": 0.25, "unheld": 0.0}
    assert out["positron_held"] == 0.0 and out["cpt"] == {}
    assert out["total"] == 0.2 and out["planes"] == {"perception": 0.5, "touch": 0.1}
    assert O2._magic_payload("quipu-voice", "hearing")["own"]["magic"] == 0.0


def test_guidance_lists_the_sources_own_plane_shadow_tokens(monkeypatch):
    from quipu import observer_service as O2, brain_kv as bk
    props = {"tokens": [
        {"token": "dock", "plane": "perception", "lift": 0.04, "sessions": ["a", "b"], "status": "proposed",
         "realisation": "none: through the six gates, two signers and the grant",
         "pathway": {"magic": 0.3, "held": 1.0}},
        {"token": "bin", "plane": "touch", "lift": 0.02, "sessions": ["a", "b"], "pathway": {}}]}
    monkeypatch.setattr(bk, "kv_get_json", lambda k, d=None: props if k == "entirety:mesh_flow:proposals" else d)
    own = O2._own_shadow_tokens("perception")
    assert [t["token"] for t in own] == ["dock"]
    assert own[0]["magic"] == 0.3 and own[0]["sessions"] == 2 and own[0]["realisation"].startswith("none")
    assert O2._magic_payload("perceptopoly", "perception")["shadow_tokens"] == own


def test_guidance_carries_the_sources_own_deepmap_paths(monkeypatch):
    from quipu import observer_service as O2, brain_kv as bk
    rec = {"at": 3.0, "paths": [{"tokens": ["a", "b"], "plane": "perception", "gain": 1.2, "hops": 1},
                                {"tokens": ["c", "d"], "plane": "touch", "gain": 1.1, "hops": 1}],
           "by_plane": {"perception": 1, "touch": 1}, "totals": {"magic": 0.4}}
    monkeypatch.setattr(bk, "kv_get_json", lambda k, d=None: rec if k == "entirety:deepmap:latest" else d)
    out = O2._deepmap_payload("perception")
    assert [p["tokens"] for p in out["paths"]] == [["a", "b"]] and out["all_paths"] == 2
    assert out["totals"] == {"magic": 0.4} and out["paths"][0]["gain"] == 1.2
