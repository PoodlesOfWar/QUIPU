"""Unit tests for quipu.security — gVisor detection, payload bounding, and security endpoints."""

import pytest
from quipu import security, observer_service


def test_gvisor_posture_structure():
    posture = security.get_security_posture()
    assert posture["ok"] is True
    assert "gvisor_sandboxed" in posture
    assert "no_new_privileges" in posture
    assert "isolation_tier" in posture
    assert "limits" in posture
    assert posture["limits"]["max_observe_bytes"] == 512 * 1024


def test_gvisor_detection_with_env(monkeypatch):
    monkeypatch.setenv("DOCKER_RUNTIME_SANDBOX", "runsc")
    assert security.is_gvisor_sandboxed() is True

    monkeypatch.setenv("DOCKER_RUNTIME_SANDBOX", "runc")
    monkeypatch.delenv("GVISOR_SANDBOXED", raising=False)
    # Default without markers
    assert security.is_gvisor_sandboxed() in (True, False)


def test_observe_payload_valid():
    valid, err = security.validate_observe_payload({
        "source": "loadopoly-ocr",
        "text": "test document sample text",
        "confidence": 0.95,
        "meta": {"page": 1, "nested": {"key": "val"}},
    })
    assert valid is True
    assert err is None


def test_observe_payload_rejects_extra_keys():
    valid, err = security.validate_observe_payload({
        "source": "loadopoly-ocr",
        "text": "sample",
        "__malicious_injected_cmd__": "rm -rf /",
    })
    assert valid is False
    assert "Unrecognized payload keys" in err


def test_observe_payload_rejects_deep_nesting():
    # Construct deeply nested dict
    nested = {"k": "v"}
    for _ in range(10):
        nested = {"k": nested}

    valid, err = security.validate_observe_payload({
        "source": "loadopoly-ocr",
        "text": "sample",
        "meta": nested,
    })
    assert valid is False
    assert "maximum nesting depth" in err


def test_observe_payload_rejects_oversized_text():
    valid, err = security.validate_observe_payload({
        "source": "loadopoly-ocr",
        "text": "x" * (security.MAX_OBSERVE_TEXT_CHARS + 10),
    })
    assert valid is False
    assert "exceeds maximum length" in err


def test_feedback_payload_validation():
    valid, err = security.validate_feedback_payload({
        "source": "bakugo",
        "expected": "Charizard #4 1st Edition",
    })
    assert valid is True
    assert err is None

    # Missing expected
    valid, err = security.validate_feedback_payload({
        "source": "bakugo",
    })
    assert valid is False
    assert "expected" in err


def test_security_endpoint_in_observer():
    posture = security.get_security_posture()
    assert isinstance(posture, dict)
    assert posture["ok"] is True
