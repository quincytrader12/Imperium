"""Measure the regime classifier against known nulls, and write the thresholds.

Run with ``imperium calibrate``. It produces
``imperium/strategy/null_calibration.json``, which is the *only* source of
thresholds the classifier uses. Nothing in this codebase compares a statistic to
a number that a human typed.

Why this exists rather than a table of textbook critical values:

* The Lo--MacKinlay z-statistic is asymptotically N(0,1) but is **not** N(0,1)
  at a 250-bar window with an overlapping horizon of q=20. Measured here, its
  null distribution is shifted and narrower than the standard normal. Using
  +/-1.96 would therefore not give a 5% false-positive rate, and worse, the
  shift is one-sided, so the errors would not even be symmetric between the two
  regimes.
* The R/S Hurst estimator is not centred on 0.5 on short windows. Its measured
  null median is recorded here. "Below 0.45 means mean-reverting" would fire on
  a large fraction of pure noise.
* The ADF critical value depends on the lag order and sample size, and using a
  MacKinnon value for a different specification than the one implemented here
  is a silent mismatch.

The procedure:

1. Fit thresholds on the primary null (a driftless geometric random walk) at a
   stated nominal level.
2. Measure *power* against generators with a known regime.
3. Re-measure the false-positive rate on **held-out** generators that were not
   used to fit anything -- GARCH, t(4) tails, high and low volatility. A
   classifier tuned until it looks good on its own test data has learned that
   data.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from imperium import config
from imperium.strategy import statistics as st

CALIBRATION_PATH = Path(__file__).with_name("null_calibration.json")

#: Nominal two-sided false-positive rate the thresholds are fitted to.
NOMINAL_ALPHA = 0.05

Generator = Callable[[int, np.random.Generator], np.ndarray]


@dataclass
class Sample:
    """The statistics measured across many independent series."""

    vr: np.ndarray
    z: np.ndarray
    hurst: np.ndarray
    adf: np.ndarray

    @classmethod
    def collect(cls, gen: Generator, trials: int, window: int,
                rng: np.random.Generator,
                drop_session_gaps: int = 0) -> "Sample":
        """Measure the statistics exactly as the engine computes them.

        ``drop_session_gaps`` is the session length in bars. When set, the
        return spanning each session boundary is dropped before any statistic
        is computed -- which is what the engine does for equities. Calibrating
        on gap-included returns and then running on gap-excluded ones would fit
        the thresholds to a distribution the classifier never actually sees, and
        every error rate measured here would describe a different program.
        """
        vr, z, hurst, adf = [], [], [], []
        for _ in range(trials):
            prices = gen(window, rng)
            rets = st.log_returns(prices)
            if drop_session_gaps:
                # The return at index i spans bars i and i+1, so the seam after
                # bar k*session is the return at index k*session - 1.
                keep = np.ones(rets.size, dtype=bool)
                for boundary in range(drop_session_gaps, window, drop_session_gaps):
                    idx = boundary - 1
                    if 0 <= idx < keep.size:
                        keep[idx] = False
                rets = rets[keep]
            v = st.variance_ratio(rets)
            if not v.valid:
                continue
            a = st.adf(np.log(prices))
            h = st.hurst_rs(rets)
            vr.append(v.vr)
            z.append(v.z)
            hurst.append(h if math.isfinite(h) else np.nan)
            adf.append(a.stat if a.valid else np.nan)
        return cls(np.array(vr), np.array(z), np.array(hurst), np.array(adf))


def _q(values: np.ndarray, *ps: float) -> list[float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return [float("nan")] * len(ps)
    return [float(np.quantile(clean, p)) for p in ps]


def build_thresholds(null: Sample, alpha: float = NOMINAL_ALPHA) -> dict:
    """Fit every bound from the measured null distribution."""
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    z_lo, z_med, z_hi = _q(null.z, lo, 0.5, hi)
    # The median of |z| under the null. The brief's reference value of 0.954 is
    # the median of |N(0,1)|; since this z is measurably not N(0,1) here, the
    # measured value is used instead. Centring a sub-threshold tilt on the wrong
    # constant makes it read non-zero on pure noise and quietly adds exposure to
    # random walks.
    abs_z_med = float(np.median(np.abs(null.z[np.isfinite(null.z)])))
    h_lo, h_med, h_hi = _q(null.hurst, lo, 0.5, hi)
    h_q1, h_q3 = _q(null.hurst, 0.25, 0.75)
    a_lo, a_med = _q(null.adf, alpha, 0.5)
    return {
        "nominal_alpha": alpha,
        "vr_z": {
            "reject_trend_above": z_hi,
            "reject_revert_below": z_lo,
            "null_median": z_med,
            "null_median_abs": abs_z_med,
            "standard_normal_would_have_used": 1.959963985,
        },
        "hurst": {
            "null_median": h_med,
            "null_q1": h_q1,
            "null_q3": h_q3,
            "trend_above": h_hi,
            "revert_below": h_lo,
            "textbook_would_have_used": 0.5,
        },
        "adf": {
            "stationary_below": a_lo,
            "null_median": a_med,
        },
    }


def _classify_counts(sample: Sample, thresholds: dict) -> dict[str, float]:
    """Apply the fitted classifier and count outcomes.

    Imported lazily so calibration can run before the regime module is written
    against a calibration file that does not yet exist.
    """
    from imperium.strategy.regime import Regime, classify_from_stats

    counts = {r.value: 0 for r in Regime}
    n = sample.z.size
    for i in range(n):
        verdict = classify_from_stats(
            vr=float(sample.vr[i]), z=float(sample.z[i]),
            hurst=float(sample.hurst[i]), adf_stat=float(sample.adf[i]),
            thresholds=thresholds,
        )
        counts[verdict.regime.value] += 1
    return {k: v / n if n else 0.0 for k, v in counts.items()}


def _sub_threshold_information(thresholds: dict, trials: int, window: int,
                               rng: np.random.Generator) -> dict:
    """Measure whether sub-threshold evidence carries any information at all.

    The brief's rule: a tilt from evidence that did not clear the bar is only
    permissible if the sign of VR-1 among those windows beats 50% against a
    known regime. This measures exactly that, on series whose regime is known by
    construction, and the classifier refuses to use a sub-threshold tilt if the
    measured accuracy does not clear the bar recorded here.
    """
    from imperium.strategy.regime import Regime, classify_from_stats

    hits = 0
    total = 0
    for gen, trending in ((st.momentum_process, True), (st.ou_process, False),
                          (st.trend_process, True)):
        for _ in range(trials):
            prices = gen(window, rng)
            rets = st.log_returns(prices)
            v = st.variance_ratio(rets)
            if not v.valid:
                continue
            a = st.adf(np.log(prices))
            verdict = classify_from_stats(
                vr=v.vr, z=v.z, hurst=st.hurst_rs(rets),
                adf_stat=a.stat if a.valid else float("nan"),
                thresholds=thresholds,
            )
            if verdict.regime is not Regime.INDETERMINATE:
                continue
            total += 1
            # Centre on the null median, not on 1.0: VR is biased below 1 at
            # this sample size, so "vr > 1 means trending" would be wrong.
            leans_trending = v.z > thresholds["vr_z"]["null_median"]
            if leans_trending == trending:
                hits += 1
    accuracy = hits / total if total else 0.0
    return {
        "samples": total,
        "directional_accuracy": accuracy,
        # A coin flip carries no information. Require a measurable margin over
        # it before a sub-threshold reading is allowed to move any money.
        "usable": bool(total >= 200 and accuracy >= 0.55),
        "required_accuracy": 0.55,
    }



def _power_curve(thresholds: dict, trials: int, window: int,
                 rng: np.random.Generator) -> dict:
    """Detection rate as a function of effect size.

    Reported because a single power figure at one arbitrary parameter value is a
    point on a curve, and the curve is the actual answer. It also makes the
    classifier's blind spot legible -- see ``drift_is_not_serial_correlation``
    in the returned dict.
    """
    from imperium.strategy.regime import Regime, classify_from_stats

    def rate(gen: Generator, want: Regime) -> float:
        hits = 0
        seen = 0
        for _ in range(trials):
            prices = gen(window, rng)
            rets = st.log_returns(prices)
            v = st.variance_ratio(rets)
            if not v.valid:
                continue
            a = st.adf(np.log(prices))
            seen += 1
            verdict = classify_from_stats(
                vr=v.vr, z=v.z, hurst=st.hurst_rs(rets),
                adf_stat=a.stat if a.valid else float("nan"),
                thresholds=thresholds,
            )
            if verdict.regime is want:
                hits += 1
        return hits / seen if seen else 0.0

    curve: dict = {"mean_reversion_ou_theta": {}, "momentum_ar1_phi": {}}
    for theta in (0.02, 0.06, 0.12, 0.25, 0.50):
        curve["mean_reversion_ou_theta"][f"{theta:.2f}"] = rate(
            lambda n, r, t=theta: st.ou_process(n, r, theta=t), Regime.MEAN_REVERTING
        )
    for phi in (0.05, 0.10, 0.18, 0.30, 0.45):
        curve["momentum_ar1_phi"][f"{phi:.2f}"] = rate(
            lambda n, r, p=phi: st.momentum_process(n, r, phi=p), Regime.TRENDING
        )
    curve["drift_is_not_serial_correlation"] = (
        "A random walk with drift has independent increments, so its variance "
        "ratio is 1 by construction and this classifier is blind to it *by "
        "design*. Detection of 'trend_with_drift' at the false-positive rate is "
        "the correct result, not a defect: drift is exploited by the momentum "
        "signal itself, while the classifier's job is to decide whether serial "
        "correlation favours momentum or mean reversion."
    )
    return curve


#: Which generators stand in for each asset class. Crypto is an unbroken walk;
#: equities are sessions separated by overnight gaps, with a U-shaped intraday
#: volatility profile held out to check the fit generalises to structure it was
#: not fitted on.
CLASS_NULLS: dict[str, Generator] = {
    "crypto": st.random_walk,
    "us_equity": lambda n, r: st.equity_session_walk(n, r, bars_per_session=100),
}

CLASS_HELD_OUT: dict[str, dict[str, Generator]] = {
    "crypto": {
        "garch_1_1": st.garch_process,
        "student_t4": st.fat_tail_process,
        "high_vol_walk": lambda n, r: st.random_walk(n, r, sigma=0.04),
        "low_vol_walk": lambda n, r: st.random_walk(n, r, sigma=0.002),
    },
    "us_equity": {
        "intraday_u_shape": lambda n, r: st.intraday_u_shape_walk(
            n, r, bars_per_session=100),
        "garch_1_1": st.garch_process,
        "student_t4": st.fat_tail_process,
        "wide_gap_sessions": lambda n, r: st.equity_session_walk(
            n, r, bars_per_session=100, gap_sigma=0.015),
    },
}


#: Bars per session used when fitting the equity class. A 250-bar window is
#: less than one 390-bar cash session, so a shorter session is used to make the
#: window straddle a seam -- which is the case the gap handling exists for and
#: the one a fit must therefore cover.
EQUITY_SESSION_BARS = 100


def run_for_class(asset_class: str, trials: int, seed: int, window: int) -> dict:
    """Fit and score the classifier for one asset class."""
    rng = np.random.default_rng(seed)
    null_gen = CLASS_NULLS[asset_class]
    drop = EQUITY_SESSION_BARS if asset_class == "us_equity" else 0
    null = Sample.collect(null_gen, trials, window, rng, drop_session_gaps=drop)
    thresholds = build_thresholds(null)

    report: dict = {
        "asset_class": asset_class,
        "seed": seed,
        "trials": trials,
        "window": window,
        "null_generator": getattr(null_gen, "__name__", str(null_gen)),
        "thresholds": thresholds,
        "null_distribution": {
            stat: dict(zip(("p2.5", "p50", "p97.5"),
                           _q(getattr(null, stat), 0.025, 0.5, 0.975)))
            for stat in ("vr", "z", "hurst", "adf")
        },
    }
    report["session_bars_dropped"] = drop
    report["false_positive_rate"] = {
        asset_class + "_null": _classify_counts(null, thresholds)
    }
    report["power"] = {}
    for name, gen in (("ou_mean_reverting", st.ou_process),
                      ("momentum_ar1", st.momentum_process)):
        report["power"][name] = _classify_counts(
            Sample.collect(gen, trials, window, rng, drop_session_gaps=drop),
            thresholds)
    report["held_out_false_positive_rate"] = {
        name: _classify_counts(
            Sample.collect(gen, trials, window, rng, drop_session_gaps=drop),
            thresholds)
        for name, gen in CLASS_HELD_OUT[asset_class].items()
    }
    sub = _sub_threshold_information(thresholds, max(200, trials // 2), window, rng)
    report["sub_threshold_evidence"] = sub
    thresholds["sub_threshold"] = sub
    return report


def run(trials: int = 1000, seed: int = 20240517, window: int = 250) -> dict:
    rng = np.random.default_rng(seed)

    null = Sample.collect(st.random_walk, trials, window, rng)
    thresholds = build_thresholds(null)

    report: dict = {
        "generated_by": "imperium.strategy.calibration",
        "seed": seed,
        "trials": trials,
        "window": window,
        "q": st.variance_ratio_horizon(window - 1),
        "thresholds": thresholds,
        "null_distribution": {
            "vr": dict(zip(("p2.5", "p25", "p50", "p75", "p97.5"),
                           _q(null.vr, 0.025, 0.25, 0.5, 0.75, 0.975))),
            "z": dict(zip(("p2.5", "p25", "p50", "p75", "p97.5"),
                          _q(null.z, 0.025, 0.25, 0.5, 0.75, 0.975))),
            "hurst": dict(zip(("p2.5", "p25", "p50", "p75", "p97.5"),
                              _q(null.hurst, 0.025, 0.25, 0.5, 0.75, 0.975))),
            "adf": dict(zip(("p2.5", "p25", "p50", "p75", "p97.5"),
                            _q(null.adf, 0.025, 0.25, 0.5, 0.75, 0.975))),
        },
    }

    # -- false positives on the fitted null, and power on known regimes --
    report["false_positive_rate"] = {"random_walk": _classify_counts(null, thresholds)}
    report["power"] = {}
    for name, gen in (("ou_mean_reverting", st.ou_process),
                      ("trend_with_drift", st.trend_process),
                      ("momentum_ar1", st.momentum_process)):
        sample = Sample.collect(gen, trials, window, rng)
        report["power"][name] = _classify_counts(sample, thresholds)

    # -- held-out nulls, not used to fit anything ------------------------
    held_out: dict[str, Generator] = {
        "garch_1_1": st.garch_process,
        "student_t4": st.fat_tail_process,
        "high_vol_walk": lambda n, r: st.random_walk(n, r, sigma=0.04),
        "low_vol_walk": lambda n, r: st.random_walk(n, r, sigma=0.002),
    }
    report["held_out_false_positive_rate"] = {}
    for name, gen in held_out.items():
        sample = Sample.collect(gen, trials, window, rng)
        report["held_out_false_positive_rate"][name] = _classify_counts(sample, thresholds)

    # Measured *after* the bars are fitted, then written into the thresholds
    # themselves -- the classifier reads its permission to tilt from the same
    # file that records the measurement granting it.
    sub = _sub_threshold_information(thresholds, max(200, trials // 2), window, rng)
    report["sub_threshold_evidence"] = sub
    thresholds["sub_threshold"] = sub

    # Power across a range of effect sizes. A single power number at one
    # arbitrary parameter value says almost nothing: it is a point on a curve
    # whose shape is the actual answer.
    report["power_curve"] = _power_curve(thresholds, max(300, trials // 3), window, rng)

    # Per-class calibrations. The crypto entry reuses the fit above; equities
    # get their own, fitted on a session-and-gap null, because a threshold
    # fitted on an unbroken walk does not describe an equity minute series.
    report["by_asset_class"] = {
        "crypto": {"thresholds": thresholds,
                   "null_generator": "random_walk",
                   "false_positive_rate": report["false_positive_rate"],
                   "held_out_false_positive_rate":
                       report["held_out_false_positive_rate"],
                   "sub_threshold_evidence": sub},
        "us_equity": run_for_class("us_equity", trials, seed + 101, window),
    }
    return report


def save(report: dict, path: Path = CALIBRATION_PATH) -> Path:
    path.write_text(json.dumps(report, indent=2), encoding=config.TEXT_ENCODING)
    return path


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main(trials: int = 1000, seed: int = 20240517, window: int = 250) -> int:
    print(f"Calibrating the regime classifier: {trials} trials, window={window}, "
          f"seed={seed}")
    print()
    report = run(trials=trials, seed=seed, window=window)
    th = report["thresholds"]

    print("MEASURED NULL (driftless geometric random walk)")
    for stat in ("vr", "z", "hurst", "adf"):
        d = report["null_distribution"][stat]
        print(f"  {stat:6} p2.5={d['p2.5']:+7.3f}  p25={d['p25']:+7.3f}  "
              f"p50={d['p50']:+7.3f}  p75={d['p75']:+7.3f}  p97.5={d['p97.5']:+7.3f}")
    print()
    print("FITTED THRESHOLDS (all measured, none typed by hand)")
    print(f"  variance-ratio z: trending above {th['vr_z']['reject_trend_above']:+.3f}, "
          f"reverting below {th['vr_z']['reject_revert_below']:+.3f}")
    print(f"    a standard-normal table would have used +/-"
          f"{th['vr_z']['standard_normal_would_have_used']:.3f} — "
          f"the measured null is not N(0,1) at this window")
    print(f"  hurst: null median {th['hurst']['null_median']:.3f} "
          f"(textbook says {th['hurst']['textbook_would_have_used']:.1f}); "
          f"corroborates trend above {th['hurst']['trend_above']:.3f}, "
          f"reversion below {th['hurst']['revert_below']:.3f}")
    print(f"  adf: stationary below {th['adf']['stationary_below']:+.3f}")
    print()
    print("FALSE-POSITIVE RATE on the null it was fitted to "
          f"(nominal {_pct(th['nominal_alpha'])})")
    for k, v in report["false_positive_rate"]["random_walk"].items():
        print(f"  {k:16} {_pct(v)}")
    print()
    print("POWER against generators with a known regime")
    for name, counts in report["power"].items():
        print(f"  {name}")
        for k, v in counts.items():
            print(f"    {k:16} {_pct(v)}")
    print()
    print("HELD-OUT NULLS — not used to fit anything. These are the honest numbers.")
    for name, counts in report["held_out_false_positive_rate"].items():
        wrong = counts.get("trending", 0.0) + counts.get("mean_reverting", 0.0)
        print(f"  {name:16} false regime calls {_pct(wrong)}   "
              f"(trend {_pct(counts.get('trending', 0))}, "
              f"revert {_pct(counts.get('mean_reverting', 0))})")
    print()
    print("POWER CURVE (detection rate vs effect size)")
    pc = report["power_curve"]
    print("  mean reversion, OU theta: " + "  ".join(
        f"{k}->{_pct(v).strip()}" for k, v in pc["mean_reversion_ou_theta"].items()))
    print("  momentum, AR(1) phi:      " + "  ".join(
        f"{k}->{_pct(v).strip()}" for k, v in pc["momentum_ar1_phi"].items()))
    print("  NOTE: " + pc["drift_is_not_serial_correlation"])
    print()
    sub = report["sub_threshold_evidence"]
    print("SUB-THRESHOLD EVIDENCE")
    print(f"  {sub['samples']} indeterminate windows against a known regime; "
          f"directional accuracy {_pct(sub['directional_accuracy'])}")
    print(f"  usable as a tilt: {sub['usable']} "
          f"(requires >= {_pct(sub['required_accuracy'])})")
    if not sub["usable"]:
        print("  -> the classifier will apply NO tilt from sub-threshold evidence,")
        print("     because it was measured not to carry information.")
    print()
    print("PER-ASSET-CLASS THRESHOLDS — the reason one strategy cannot serve both")
    for name, block in report["by_asset_class"].items():
        th = block["thresholds"]
        z, h = th["vr_z"], th["hurst"]
        print(f"  {name}")
        print(f"    variance-ratio z: trending above {z['reject_trend_above']:+.3f}, "
              f"reverting below {z['reject_revert_below']:+.3f}")
        print(f"    hurst null median {h['null_median']:.3f}, "
              f"corroborates trend above {h['trend_above']:.3f}")
        print(f"    adf stationary below {th['adf']['stationary_below']:+.3f}")
        wrong = block["held_out_false_positive_rate"]
        worst = max((v.get("trending", 0) + v.get("mean_reverting", 0)
                     for v in wrong.values()), default=0.0)
        print(f"    worst held-out false-regime rate {_pct(worst)}")
    ce = report["by_asset_class"]["crypto"]["thresholds"]["vr_z"]
    ee = report["by_asset_class"]["us_equity"]["thresholds"]["vr_z"]
    print(f"  the equity trending bar sits {ee['reject_trend_above'] - ce['reject_trend_above']:+.3f} "
          f"from crypto's; using one for the other changes the error rate directly")
    print()
    path = save(report)
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
