"""Per-endpoint readout policy (utils/endpoint_readout.py).

The policy table encodes measured head-to-head results, so these tests pin the
behaviour that those measurements chose — especially that ACS goes through the
calibrated margin (where the recovered sensitivity lives) and LVEF goes through
numeric EF parsing (where the model's own continuous estimate lives).
"""
import math

import pytest

from utils.endpoint_readout import (
    ENDPOINT_POLICY, ReadoutResult, parse_ef, parse_text_yes_no, read_endpoint,
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
    """The locked ACS threshold is 0.076, not 0.5 — that low bar is exactly what buys
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


def test_margin_endpoint_without_margin_degrades_to_text():
    r = read_endpoint("acute_coronary_occlusion", generation="Yes.", margin=None, calibration=CAL)
    assert r.answer is True and r.score is None


def test_parsers():
    assert parse_text_yes_no("Yes, atrial fibrillation") is True
    assert parse_text_yes_no("No structural heart disease") is False
    assert parse_text_yes_no("Borderline findings") is None
    assert parse_ef("the ejection fraction is 41.5%") == 41.5
    assert parse_ef("no number here") is None
