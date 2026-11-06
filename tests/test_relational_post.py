"""POST /relational keeps a coordinator's record without feeding any text to the mesh."""
from quipu import observer_service, mesh_slm
from quipu.qpsi import edge_admission


def _corpus_rows() -> int:
    with mesh_slm._conn() as cn:
        try:
            return int(cn.execute("SELECT COUNT(*) AS c FROM mesh_slm_vocab").fetchone()["c"])
        except Exception:
            return 0


def test_relational_post_keeps_record_and_learns_nothing():
    before = _corpus_rows()
    rec = {"schema": "holon.coordinator.run/1", "agent_id": "plan:metrology-calibration",
           "readout": {"control": "PASSED"}, "steps": {"stockopoly": "PASSED"}}
    code, res = observer_service._relational_post({"source": "hubcore", "relational": rec})
    assert code == 200 and res["ok"] and res["kept"] == {"entity": "plan:metrology-calibration",
                                                          "schema": "holon.coordinator.run/1"}
    assert _corpus_rows() == before                     # nothing reached the vocabulary
    code, got = observer_service._relational({"source": ["hubcore"], "entity": ["plan:metrology-calibration"]})
    assert code == 200 and got["record"]["steps"] == {"stockopoly": "PASSED"}
    assert got["record"]["source"] == "hubcore"
    code, idx = observer_service._relational({"source": ["hubcore"]})
    assert "hubcore:plan:metrology-calibration" in idx["index"]


def test_relational_post_refuses_unknown_source_and_bad_record():
    assert observer_service._relational_post({"source": "nobody", "relational": {"schema": "x", "agent_id": "y"}})[0] == 400
    assert observer_service._relational_post({"source": "hubcore", "relational": {"schema": "x"}})[0] == 400
    assert observer_service._relational_post({"source": "hubcore", "relational": "text"})[0] == 400


def test_relational_post_is_behind_the_edge():
    assert ("POST", "/relational") in edge_admission.MUTATING
