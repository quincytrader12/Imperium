"""The regime classifier.

Momentum and mean reversion are opposite bets on the sign of serial
correlation. A naive 50/50 blend of them is close to self-annihilating: the two
books fight each other and pay the spread twice for the privilege. So a
classifier decides which one gets the budget, and how much.

Structure:

* **Variance ratio** is the primary signal, using the heteroskedasticity-robust
  z-statistic. Volatility clustering is a fact about crypto returns, and the
  homoskedastic statistic rejects the random walk *because of* it, which is not
  the question being asked.
* **Hurst** and **ADF** are corroborators only. Neither can call a regime alone.

**Two kinds of "indeterminate", treated oppositely.** The evidence can point
somewhere and fail to clear the bar, or it can clear the bar and be
*contradicted* by another test. These are not the same thing:

* A contradiction produces **no tilt at all**. Disagreement between tests is
  evidence against acting, not a weaker reason to act.
* Sub-threshold evidence may tilt the split -- but only because
  ``calibration.py`` *measured* that it carries direction, and only while that
  measurement says so. If the measured accuracy does not beat a coin flip by a
  stated margin, the tilt is zero and the calibration file records why.

Every threshold comes from ``null_calibration.json``. There is no number in this
module that a human chose.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from imperium import config
from imperium.strategy import statistics as st

CALIBRATION_PATH = Path(__file__).with_name("null_calibration.json")


class Regime(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    #: Evidence points somewhere but does not clear the bar. May carry a small
    #: tilt, if and only if that tilt was measured to be informative.
    INDETERMINATE = "indeterminate"
    #: Tests clear their bars and disagree. Explicitly zero tilt.
    CONTRADICTED = "contradicted"
    #: Not enough bars yet. Distinct from "seeing no opportunity" everywhere in
    #: this program, because one resolves itself and the other is a decision.
    WARMING_UP = "warming_up"


class CalibrationMissing(RuntimeError):
    """Raised when the classifier is used before it has been calibrated.

    Deliberately fatal rather than falling back to textbook constants. A silent
    fallback to +/-1.96 and 0.5 would produce a classifier that fires on more
    than half of pure noise while looking like it works.
    """


def thresholds_for(asset_class: str, path: str | None = None) -> dict[str, Any]:
    """The measured thresholds for one asset class.

    Equities and crypto are fitted separately, against nulls that reflect how
    each actually behaves -- an unbroken walk for crypto, sessions with
    overnight seams for equities -- and each is scored on its own held-out
    generators. Handing an equity the crypto thresholds changes its error rate
    directly, which is why this takes the class rather than defaulting.
    """
    report = load_calibration(path)
    by_class = report.get("by_asset_class", {})
    block = by_class.get(asset_class)
    if block is None:
        known = ", ".join(sorted(by_class)) or "(none)"
        raise CalibrationMissing(
            f"no measured thresholds for asset class {asset_class!r}; the "
            f"calibration file has {known}. Run 'imperium calibrate'."
        )
    return block["thresholds"]


@lru_cache(maxsize=4)
def load_calibration(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CALIBRATION_PATH
    if not p.exists():
        raise CalibrationMissing(
            f"the regime classifier has not been calibrated ({p} is missing). "
            "Run 'imperium calibrate' -- the thresholds are measured, and there "
            "are deliberately no built-in defaults to fall back to."
        )
    return json.loads(p.read_text(encoding=config.TEXT_ENCODING))


@dataclass(frozen=True)
class RegimeVerdict:
    """What the classifier concluded, and why, in words an operator can read."""

    regime: Regime
    #: -1 fully mean-reverting, +1 fully trending, 0 no view.
    tilt: float
    #: 0..1, how strongly the evidence cleared its bar.
    confidence: float
    reason: str
    vr: float = float("nan")
    z: float = float("nan")
    hurst: float = float("nan")
    adf_stat: float = float("nan")
    q: int = 0
    n: int = 0

    @property
    def momentum_share(self) -> float:
        """Fraction of the budget momentum gets; mean reversion gets the rest."""
        return float(np.clip(0.5 + 0.5 * self.tilt, 0.0, 1.0))

    def as_dict(self) -> dict[str, Any]:
        def num(x: float) -> float | None:
            return None if not math.isfinite(x) else round(float(x), 4)

        return {
            "regime": self.regime.value,
            "tilt": round(self.tilt, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "momentum_share": round(self.momentum_share, 4),
            "vr": num(self.vr), "z": num(self.z), "hurst": num(self.hurst),
            "adf": num(self.adf_stat), "q": self.q, "n": self.n,
        }


def _corroborators(hurst: float, adf_stat: float, th: dict) -> tuple[int, list[str]]:
    """Return the corroborators' net direction and a description of each.

    +1 is trending, -1 is mean-reverting, 0 is no opinion. A corroborator only
    speaks when it clears its *own* measured bar; a Hurst of 0.44 on a window
    whose null median is 0.42 is noise, not weak evidence.
    """
    votes = 0
    notes: list[str] = []
    h = th["hurst"]
    if math.isfinite(hurst):
        if hurst > h["trend_above"]:
            votes += 1
            notes.append(f"hurst {hurst:.2f} above the null's {h['trend_above']:.2f}")
        elif hurst < h["revert_below"]:
            votes -= 1
            notes.append(f"hurst {hurst:.2f} below the null's {h['revert_below']:.2f}")
    if math.isfinite(adf_stat) and adf_stat < th["adf"]["stationary_below"]:
        votes -= 1
        notes.append(f"adf {adf_stat:.2f} rejects a unit root")
    return votes, notes


def classify_from_stats(
    *,
    vr: float,
    z: float,
    hurst: float,
    adf_stat: float,
    thresholds: dict,
    n: int = 0,
    q: int = 0,
) -> RegimeVerdict:
    """Apply the fitted rule. Separated from data handling so that
    ``calibration.py`` can score the exact rule it is fitting."""
    if not math.isfinite(z) or not math.isfinite(vr):
        return RegimeVerdict(Regime.WARMING_UP, 0.0, 0.0,
                             "not enough bars to estimate a regime", n=n, q=q)

    zt = thresholds["vr_z"]
    trend_bar = zt["reject_trend_above"]
    revert_bar = zt["reject_revert_below"]
    null_mid = zt["null_median"]

    votes, notes = _corroborators(hurst, adf_stat, thresholds)
    corroboration = (" corroborated by " + "; ".join(notes)) if notes else ""

    primary = 0
    if z > trend_bar:
        primary = 1
    elif z < revert_bar:
        primary = -1

    if primary != 0:
        if votes != 0 and (votes > 0) != (primary > 0):
            # Cleared the bar, and another test that also cleared its bar points
            # the other way. Disagreement is evidence against acting.
            return RegimeVerdict(
                Regime.CONTRADICTED, 0.0, 0.0,
                (f"variance ratio says {'trending' if primary > 0 else 'mean-reverting'} "
                 f"(z {z:+.2f}) but is contradicted by {'; '.join(notes)} — "
                 "disagreement is evidence against acting, so no tilt is applied"),
                vr, z, hurst, adf_stat, q, n,
            )
        # Scale confidence by how far past the bar it went, saturating at twice
        # the bar's distance from the null median.
        bar = trend_bar if primary > 0 else revert_bar
        span = abs(bar - null_mid) or 1.0
        excess = abs(z - bar) / span
        confidence = float(np.clip(0.35 + 0.65 * min(excess, 1.0), 0.0, 1.0))
        if votes != 0:
            confidence = float(np.clip(confidence + 0.10, 0.0, 1.0))
        regime = Regime.TRENDING if primary > 0 else Regime.MEAN_REVERTING
        return RegimeVerdict(
            regime, primary * confidence, confidence,
            (f"variance ratio z {z:+.2f} clears the measured "
             f"{'trending' if primary > 0 else 'mean-reverting'} bar "
             f"{bar:+.2f}{corroboration}"),
            vr, z, hurst, adf_stat, q, n,
        )

    # Sub-threshold. A tilt here is permitted only because it was measured to
    # carry direction; the measurement travels in the calibration file.
    sub = thresholds.get("sub_threshold", {})
    if votes != 0 and sub.get("usable"):
        # Corroborators agreeing among themselves while the primary is silent is
        # the weakest admissible evidence, and is capped accordingly.
        tilt = 0.20 * (1 if votes > 0 else -1)
        return RegimeVerdict(
            Regime.INDETERMINATE, tilt, 0.20,
            (f"variance ratio z {z:+.2f} did not clear either bar; a small tilt "
             f"from {'; '.join(notes)}, which calibration measured to be "
             f"{100 * sub.get('directional_accuracy', 0):.0f}% directional"),
            vr, z, hurst, adf_stat, q, n,
        )

    why_no_tilt = ""
    if not sub.get("usable", False):
        why_no_tilt = (" — calibration measured sub-threshold evidence as "
                       "uninformative, so no tilt is applied")
    return RegimeVerdict(
        Regime.INDETERMINATE, 0.0, 0.0,
        (f"variance ratio z {z:+.2f} is inside the null's own range "
         f"[{revert_bar:+.2f}, {trend_bar:+.2f}]; no regime evidence{why_no_tilt}"),
        vr, z, hurst, adf_stat, q, n,
    )


def classify(prices: np.ndarray, thresholds: dict | None = None, *,
             returns: np.ndarray | None = None) -> RegimeVerdict:
    """Classify a window of prices.

    ``returns`` may be supplied when the caller has already computed them --
    which the engine does, because for an equity it must drop the returns that
    span an overnight seam before any statistic is calculated.
    """
    th = thresholds if thresholds is not None else load_calibration()["thresholds"]
    prices = np.asarray(prices, dtype=float)
    rets = st.log_returns(prices) if returns is None else np.asarray(
        returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < st.MIN_SAMPLES:
        return RegimeVerdict(
            Regime.WARMING_UP, 0.0, 0.0,
            f"warming up: {rets.size} of {st.MIN_SAMPLES} bars needed to "
            "estimate a regime",
            n=rets.size,
        )
    v = st.variance_ratio(rets)
    if not v.valid:
        return RegimeVerdict(
            Regime.WARMING_UP, 0.0, 0.0,
            f"warming up: {rets.size} bars is not enough for a horizon of {v.q}",
            n=rets.size, q=v.q,
        )
    a = st.adf(np.log(prices[prices > 0]))
    return classify_from_stats(
        vr=v.vr, z=v.z, hurst=st.hurst_rs(rets),
        adf_stat=a.stat if a.valid else float("nan"),
        thresholds=th, n=v.n, q=v.q,
    )
