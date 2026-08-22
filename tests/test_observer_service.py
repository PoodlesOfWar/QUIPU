"""Tests for quipu.observer_service — endpoints, source routing, and /slm inference."""

import json
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
