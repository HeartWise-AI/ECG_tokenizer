"""Per-endpoint readout policy (utils/endpoint_readout.py).

The policy table encodes measured head-to-head results, so these tests pin the
behaviour that those measurements chose - especially that ACS goes through the
calibrated margin (where the recovered sensitivity lives) and LVEF goes through
numeric EF parsing (where the model's own continuous estimate lives).
"""
import json
import math

import pytest

import utils.endpoint_readout as endpoint_readout
from utils.artifact_provenance import atomic_write_json, file_identity
from utils.endpoint_contract import (
    endpoint_contract_sha256,
    endpoint_implementation_identity,
)
from utils.endpoint_readout import (
    ENDPOINT_POLICY, parse_ef, parse_text_yes_no, read_endpoint,
)

CAL = {
    "acute_coronary_occlusion": {"a": 0.9964, "b": -3.1165, "threshold": 0.0763},
    "lvef_lte_40": {"a": 0.7893, "b": -4.9759, "threshold": 0.1519},
    "lvef_lt_50": {"a": 0.7650, "b": -4.0447, "threshold": 0.2720},
    "incident_afib_5y": {"a": 0.3868, "b": -3.9672, "threshold": 0.3527},
    "shd": {"a": 0.8966, "b": -2.4015, "threshold": 0.4601},
}


def test_unknown_endpoint_rejected():
    with pytest.raises(KeyError, match="no measured policy"):
        read_endpoint("brugada", generation="Yes.", margin=1.0)


def test_policy_table_matches_the_measurements_we_made():
    assert ENDPOINT_POLICY["acute_coronary_occlusion"]["policy"] == "margin"
    assert ENDPOINT_POLICY["lvef_lte_40"]["policy"] == "numeric"
    assert ENDPOINT_POLICY["incident_afib_5y"]["policy"] == "text"
    assert ENDPOINT_POLICY["shd"]["policy"] == "text"
    # every endpoint records why, so the choice stays auditable after a retrain
    assert all(p["measured"] for p in ENDPOINT_POLICY.values())


def test_acs_threshold_sits_well_below_a_naive_half():
    """The locked ACS threshold is 0.076, not 0.5 - that low bar is exactly what buys
    back sensitivity, so pin it. margin=0 must still read No (prob 0.042)."""
    cal = CAL["acute_coronary_occlusion"]
    assert cal["threshold"] < 0.1
    r = read_endpoint("acute_coronary_occlusion",
                      generation="No acute coronary occlusion.", margin=0.0, calibration=CAL)
    prob = 1 / (1 + math.exp(-(cal["a"] * 0.0 + cal["b"])))
    assert r.policy == "margin"
    assert r.score == pytest.approx(prob, abs=1e-9)
    assert prob < cal["threshold"]
    assert r.answer is False and r.text_answer is False and r.disagrees is False


def test_acs_margin_overrides_a_conservative_text_answer():
    """The whole point: text says No, a modest positive margin says Yes, margin wins.
    Crossing point is margin ~0.63 (prob 0.0763), far below the naive 0.5 boundary."""
    r = read_endpoint("acute_coronary_occlusion",
                      generation="No acute coronary occlusion.", margin=1.0, calibration=CAL)
    assert r.policy == "margin"
    assert r.answer is True and r.text_answer is False and r.disagrees is True


def test_acs_clearly_positive_margin_answers_yes():
    r = read_endpoint("acute_coronary_occlusion", generation="No.", margin=4.0, calibration=CAL)
    assert r.answer is True and r.text_answer is False and r.disagrees is True


@pytest.mark.parametrize(("margin", "expected"), [(-1e6, False), (1e6, True)])
def test_margin_readout_handles_extreme_finite_logits(margin, expected):
    r = read_endpoint(
        "acute_coronary_occlusion",
        generation="Borderline findings.",
        margin=margin,
        calibration=CAL,
    )

    assert r.answer is expected
    assert r.score in (0.0, 1.0)


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), float("-inf")])
def test_margin_readout_rejects_non_finite_margin(margin):
    with pytest.raises(ValueError, match="margin must be finite"):
        read_endpoint(
            "acute_coronary_occlusion",
            generation="Borderline findings.",
            margin=margin,
            calibration=CAL,
        )


@pytest.mark.parametrize(
    "entry",
    [
        {"a": float("nan"), "b": 0.0, "threshold": 0.5},
        {"a": 1.0, "b": float("inf"), "threshold": 0.5},
        {"a": 1.0, "b": 0.0, "threshold": 1.1},
    ],
)
def test_margin_readout_rejects_invalid_calibration(entry):
    calibration = {**CAL, "acute_coronary_occlusion": entry}
    with pytest.raises(ValueError, match="calibration"):
        read_endpoint(
            "acute_coronary_occlusion",
            generation="Borderline findings.",
            margin=0.0,
            calibration=calibration,
        )


def test_lvef_uses_parsed_number_not_yes_no():
    r = read_endpoint("lvef_lte_40",
                      generation="The ejection fraction is 32%.", margin=-5.0, calibration=CAL)
    assert r.policy == "numeric" and r.score == 32.0 and r.answer is True
    r2 = read_endpoint("lvef_lte_40",
                       generation="The ejection fraction is 55%.", margin=5.0, calibration=CAL)
    assert r2.answer is False


def test_lvef_falls_back_to_margin_when_no_number_present():
    r = read_endpoint("lvef_lte_40", generation="Reduced systolic function.",
                      margin=6.0, calibration=CAL)
    assert r.policy == "margin" and r.score is not None


def test_afib_and_shd_keep_the_generated_text():
    for ep in ("incident_afib_5y", "shd"):
        r = read_endpoint(ep, generation="Yes, elevated risk.", margin=-9.0, calibration=CAL)
        assert r.policy == "text" and r.answer is True and r.disagrees is False


def test_afib_text_policy_accepts_prompt_contract_positive_phrase():
    r = read_endpoint(
        "incident_afib_5y",
        generation="Elevated risk of new-onset atrial fibrillation within five years.",
        margin=None,
        calibration=CAL,
    )

    assert r.policy == "text"
    assert r.answer is True


def test_margin_endpoint_without_margin_fails_closed():
    with pytest.raises(ValueError, match="margin is required"):
        read_endpoint(
            "acute_coronary_occlusion",
            generation="Yes.",
            margin=None,
            calibration=CAL,
        )


def test_numeric_endpoint_without_number_or_margin_fails_closed():
    with pytest.raises(ValueError, match="margin is required"):
        read_endpoint(
            "lvef_lte_40",
            generation="Reduced systolic function.",
            margin=None,
            calibration=CAL,
        )


def test_parsers():
    assert parse_text_yes_no("Yes, atrial fibrillation") is True
    assert parse_text_yes_no("No structural heart disease") is False
    assert parse_text_yes_no("Structural heart disease is not present") is False
    assert parse_text_yes_no("No high risk features are present") is False
    assert parse_text_yes_no("There are no high risk features present") is False
    assert parse_text_yes_no("There is no structural heart disease present") is False
    assert parse_text_yes_no("There is no acute coronary occlusion present") is False
    assert parse_text_yes_no("Normal sinus rhythm") is None
    assert parse_text_yes_no("Yesterday the risk was discussed") is None
    assert parse_text_yes_no("Borderline findings") is None
    assert parse_ef("the ejection fraction is 41.5%") == 41.5
    assert parse_ef("the ejection fraction is 0%") == 0.0
    assert parse_ef("the ejection fraction is 100%") == 100.0
    assert parse_ef("the ejection fraction is 101%") is None
    assert parse_ef("the ejection fraction is 999999%") is None
    assert parse_ef("no number here") is None


@pytest.mark.parametrize(
    "text",
    [
        "There is no question that structural heart disease is present.",
        "Without doubt, acute coronary occlusion is present.",
        "Structural heart disease is not only present, but severe.",
    ],
)
def test_parser_does_not_treat_positive_idioms_as_negation(text):
    assert parse_text_yes_no(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "There is no evidence of structural heart disease.",
        "The ECG does not indicate acute coronary occlusion.",
        "There is no evidence to suggest acute coronary occlusion.",
        "There are no findings to support structural heart disease.",
        "Acute coronary occlusion is not present.",
        "Structural heart disease appears unlikely.",
    ],
)
def test_parser_recognizes_grammatical_negation(text):
    assert parse_text_yes_no(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "The patient has a low risk of atrial fibrillation within five years.",
        "There is a low likelihood of atrial fibrillation within five years.",
        "The risk of atrial fibrillation within five years is low.",
        "Atrial fibrillation is possible but unlikely.",
    ],
)
def test_parser_recognizes_low_afib_risk_as_negative(text):
    assert parse_text_yes_no(text, endpoint="incident_afib_5y") is False


def test_parser_fails_closed_for_mixed_endpoint_clauses():
    text = "No structural heart disease, but acute coronary occlusion is present."

    assert parse_text_yes_no(text) is None
    assert parse_text_yes_no(text, endpoint="shd") is False
    assert parse_text_yes_no(text, endpoint="acute_coronary_occlusion") is True


def _write_bound_calibration(tmp_path, monkeypatch, checkpoint):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    margins = tmp_path / "margins.npz"
    script = tmp_path / "calibrate.py"
    for path in (train, test, margins, script):
        path.write_bytes(path.name.encode("utf-8"))
    checkpoint_identity = file_identity(checkpoint)
    test_identity = file_identity(test)
    contract_hash = endpoint_contract_sha256(ENDPOINT_POLICY)
    document = {
        "provenance": {
            "schema_version": 2,
            "kind": "endpoint_train_calibration",
            "checkpoint": checkpoint_identity,
            "train_parquet": file_identity(train),
            "test_parquet": test_identity,
            "test_margins": file_identity(margins),
            "script": file_identity(script),
            "endpoint_contract_sha256": contract_hash,
            "test_margin_cache_provenance": {
                "semantic": {
                "checkpoint": checkpoint_identity,
                "parquet": test_identity,
                "endpoint_contract_sha256": contract_hash,
                "implementation": endpoint_implementation_identity(),
            }
            },
        },
        "endpoints": {
            name: {
                "platt_a": values["a"],
                "platt_b": values["b"],
                "locked_threshold_prob": values["threshold"],
            }
            for name, values in CAL.items()
        },
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(document), encoding="utf-8")
    markdown_path = tmp_path / "endpoint_readout_train_calibrated.md"
    margins_path = tmp_path / "train_calib_margins.npz"
    markdown_path.write_text("# calibration\n", encoding="utf-8")
    margins_path.write_bytes(b"test margins")
    atomic_write_json(
        tmp_path / "endpoint_readout_train_calibrated.bundle.json",
        {
            "schema_version": 1,
            "kind": "artifact_bundle",
            "provenance": document["provenance"],
            "files": {
                calibration_path.name: file_identity(calibration_path),
                markdown_path.name: file_identity(markdown_path),
                margins_path.name: file_identity(margins_path),
            },
        },
    )
    monkeypatch.setattr(endpoint_readout, "CALIBRATION_JSON", calibration_path)
    return calibration_path


def test_locked_calibration_accepts_exact_checkpoint_and_prompt_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    _write_bound_calibration(tmp_path, monkeypatch, checkpoint)

    result = read_endpoint(
        "acute_coronary_occlusion",
        generation="No acute coronary occlusion.",
        margin=1.0,
        checkpoint_path=checkpoint,
    )

    assert result.answer is True


def test_locked_calibration_rejects_changed_checkpoint_bytes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    _write_bound_calibration(tmp_path, monkeypatch, checkpoint)
    checkpoint.write_bytes(b"checkpoint-b")

    with pytest.raises(ValueError, match="checkpoint does not match"):
        read_endpoint(
            "acute_coronary_occlusion",
            generation="No acute coronary occlusion.",
            margin=1.0,
            checkpoint_path=checkpoint,
        )


def test_locked_calibration_rejects_changed_inference_implementation(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    _write_bound_calibration(tmp_path, monkeypatch, checkpoint)
    monkeypatch.setattr(
        endpoint_readout,
        "endpoint_implementation_identity",
        lambda: {"sha256": "changed"},
    )

    with pytest.raises(ValueError, match="not bound"):
        read_endpoint(
            "acute_coronary_occlusion",
            generation="No acute coronary occlusion.",
            margin=1.0,
            checkpoint_path=checkpoint,
        )


def test_locked_calibration_rejects_member_replacement_after_bundle_validation(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    calibration_path = _write_bound_calibration(tmp_path, monkeypatch, checkpoint)
    validate_bundle = endpoint_readout.validate_artifact_bundle
    swapped = False

    def validate_then_replace(*args, **kwargs):
        nonlocal swapped
        manifest = validate_bundle(*args, **kwargs)
        if not swapped:
            swapped = True
            document = json.loads(calibration_path.read_text(encoding="utf-8"))
            document["endpoints"]["acute_coronary_occlusion"]["platt_a"] = 99.0
            calibration_path.write_text(json.dumps(document), encoding="utf-8")
        return manifest

    monkeypatch.setattr(
        endpoint_readout,
        "validate_artifact_bundle",
        validate_then_replace,
    )

    with pytest.raises(ValueError, match="changed during committed read"):
        endpoint_readout._load_calibration(checkpoint)
