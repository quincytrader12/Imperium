"""The regime classifier and its calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from godalgo.strategy import statistics as st
from godalgo.strategy.calibration import CALIBRATION_PATH, build_thresholds, Sample
from godalgo.strategy.regime import (
    CalibrationMissing, Regime, classify, classify_from_stats, load_calibration,
)

SEED = 20240517


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return load_calibration()["thresholds"]


def test_the_shipped_calibration_exists_and_records_its_seed():
    """Prevents: a classifier whose thresholds cannot be reproduced. A measured
    threshold with no stated seed and trial count is not a measurement."""
    report = load_calibration()
    assert report["seed"] and report["trials"] >= 1000
    assert report["window"] == 250


def test_the_classifier_refuses_to_run_uncalibrated(tmp_path):
    """Prevents: a silent fallback to textbook constants. Falling back to +/-1.96
    and a Hurst of 0.5 would produce a classifier that fires on a large fraction
    of pure noise while appearing to work."""
    load_calibration.cache_clear()
    with pytest.raises(CalibrationMissing, match="godalgo calibrate"):
        load_calibration(str(tmp_path / "absent.json"))


def test_the_hurst_null_is_not_centred_on_one_half(thresholds):
    """Prevents: the exact error the brief warns about. The R/S estimator is not
    centred on 0.5 in small samples, so a threshold written as 'below 0.45 means
    mean-reverting' fires on a large share of pure noise.

    The measured median for *this* estimator at this window is recorded in the
    calibration file. Note it differs from the 0.41-0.43 the brief cites for a
    different R/S variant, which is precisely why the number is measured here
    rather than copied."""
    h = thresholds["hurst"]
    assert abs(h["null_median"] - 0.5) > 0.05, (
        "if the null median really is 0.5 for this estimator, the calibration "
        "is not measuring what it claims to"
    )
    assert h["revert_below"] < h["null_median"] < h["trend_above"]


def test_the_variance_ratio_z_null_is_not_a_standard_normal(thresholds):
    """Prevents: using +/-1.96 from a table. The Lo-MacKinlay z is only
    asymptotically N(0,1); at a 250-bar window with q=20 the measured null is
    shifted and asymmetric, so a symmetric table threshold gives neither the
    nominal error rate nor equal error rates between the two regimes."""
    z = thresholds["vr_z"]
    assert z["reject_revert_below"] < 0 < z["reject_trend_above"]
    asymmetry = abs(abs(z["reject_trend_above"]) - abs(z["reject_revert_below"]))
    assert asymmetry > 0.2, (
        "the measured null looked symmetric; if that is real, the justification "
        "for not using a standard normal table needs revisiting"
    )


def test_the_false_positive_rate_matches_the_nominal_level():
    """Prevents: a classifier that calls a regime on noise. Measured against 600
    fresh random walks generated from a seed the thresholds were not fitted to."""
    rng = np.random.default_rng(SEED + 999)
    th = load_calibration()["thresholds"]
    calls = 0
    trials = 600
    for _ in range(trials):
        verdict = classify(st.random_walk(250, rng), th)
        if verdict.regime in (Regime.TRENDING, Regime.MEAN_REVERTING):
            calls += 1
    rate = calls / trials
    assert rate < 0.10, f"false-positive rate {rate:.1%} against a nominal 5%"


def test_it_has_real_power_against_a_known_mean_reverting_process():
    """Prevents: a classifier so conservative it never fires — which would have a
    perfect false-positive rate and no value. Measured against an OU process
    whose reversion is strong enough that the calibration's own power curve says
    it should be detected."""
    rng = np.random.default_rng(SEED + 555)
    th = load_calibration()["thresholds"]
    hits = 0
    trials = 300
    for _ in range(trials):
        verdict = classify(st.ou_process(250, rng, theta=0.25), th)
        if verdict.regime is Regime.MEAN_REVERTING:
            hits += 1
    assert hits / trials > 0.80, f"power {hits / trials:.1%} at theta=0.25"


def test_a_contradiction_produces_no_tilt_at_all(thresholds):
    """Prevents: treating disagreement as weak agreement. When one test clears
    its bar saying 'trending' and another clears its bar saying 'mean-reverting',
    that is evidence against acting — not a smaller reason to act."""
    verdict = classify_from_stats(
        vr=1.6,
        z=thresholds["vr_z"]["reject_trend_above"] + 1.0,     # clears: trending
        hurst=thresholds["hurst"]["revert_below"] - 0.05,     # clears: reverting
        adf_stat=thresholds["adf"]["stationary_below"] - 1.0,  # clears: stationary
        thresholds=thresholds,
    )
    assert verdict.regime is Regime.CONTRADICTED
    assert verdict.tilt == 0.0
    assert verdict.momentum_share == 0.5
    assert "disagreement is evidence against acting" in verdict.reason


def test_sub_threshold_evidence_tilts_only_because_it_was_measured(thresholds):
    """Prevents: a tilt from evidence that carries no direction, which quietly
    adds exposure to random walks. The permission to tilt travels in the
    calibration file alongside the measurement that granted it."""
    sub = thresholds.get("sub_threshold", {})
    assert "directional_accuracy" in sub and "usable" in sub

    inside_z = (thresholds["vr_z"]["reject_trend_above"]
                + thresholds["vr_z"]["reject_revert_below"]) / 2
    with_corroboration = classify_from_stats(
        vr=1.0, z=inside_z, hurst=thresholds["hurst"]["trend_above"] + 0.05,
        adf_stat=float("nan"), thresholds=thresholds,
    )
    assert with_corroboration.regime is Regime.INDETERMINATE
    if sub["usable"]:
        assert with_corroboration.tilt != 0.0
    else:
        assert with_corroboration.tilt == 0.0

    # With the permission withdrawn, the same evidence must move nothing.
    withdrawn = dict(thresholds)
    withdrawn["sub_threshold"] = {"usable": False, "directional_accuracy": 0.5}
    assert classify_from_stats(
        vr=1.0, z=inside_z, hurst=thresholds["hurst"]["trend_above"] + 0.05,
        adf_stat=float("nan"), thresholds=withdrawn,
    ).tilt == 0.0


def test_evidence_free_windows_produce_no_tilt(thresholds):
    """Prevents: a measure centred on the wrong constant reading non-zero on pure
    noise. A z sitting exactly at the null's own median must move nothing."""
    verdict = classify_from_stats(
        vr=1.0, z=thresholds["vr_z"]["null_median"], hurst=float("nan"),
        adf_stat=float("nan"), thresholds=thresholds,
    )
    assert verdict.tilt == 0.0
    assert verdict.momentum_share == 0.5


def test_warming_up_is_never_reported_as_no_opportunity():
    """Prevents: the failure mode the whole UI is built around. A bot that has
    not seen enough bars and a bot that has decided not to trade look identical
    unless they are named differently."""
    verdict = classify(np.linspace(100, 101, 20))
    assert verdict.regime is Regime.WARMING_UP
    assert "warming up" in verdict.reason.lower()


def test_the_variance_ratio_horizon_scales_with_the_window():
    """Prevents: a fixed q. The estimator's variance depends on q/n, so a
    constant horizon means the test behaves differently at different window
    lengths without anyone changing a threshold."""
    assert st.variance_ratio_horizon(120) == 10
    assert st.variance_ratio_horizon(250) == 20
    assert st.variance_ratio_horizon(48) == 4      # clipped at the floor
    assert st.variance_ratio_horizon(10_000) == 64  # clipped at the ceiling


def test_the_variance_ratio_detects_the_sign_of_serial_correlation():
    """Prevents: a sign error in the estimator, which would swap the two
    strategies and make the classifier reliably wrong rather than merely
    useless."""
    rng = np.random.default_rng(5)
    trending = np.mean([st.variance_ratio(
        st.log_returns(st.momentum_process(500, rng, phi=0.4))).vr
        for _ in range(40)])
    reverting = np.mean([st.variance_ratio(
        st.log_returns(st.ou_process(500, rng, theta=0.4))).vr
        for _ in range(40)])
    assert trending > 1.0 > reverting


def test_held_out_generators_do_not_inflate_the_error_rate():
    """Prevents: a classifier tuned until it looks good on its own test data. The
    thresholds were fitted on constant-variance Gaussian walks; these are
    martingales too, but with volatility clustering and fat tails, and were used
    to fit nothing."""
    rng = np.random.default_rng(SEED + 4242)
    th = load_calibration()["thresholds"]
    for name, gen in (("garch", st.garch_process), ("t4", st.fat_tail_process)):
        calls = 0
        trials = 300
        for _ in range(trials):
            v = classify(gen(250, rng), th)
            if v.regime in (Regime.TRENDING, Regime.MEAN_REVERTING):
                calls += 1
        assert calls / trials < 0.12, f"{name}: {calls / trials:.1%} false calls"
