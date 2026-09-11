"""Run the cluster's front-end checks under Node, from the Python suite.

The four properties the brief calls out for the orb field -- dedupe, bounded
replay, a bloom-measured budget, jittered speed and depth -- are invisible in a
screenshot and easy to regress. They are asserted in ``tests/js/cluster_checks.js``
and surfaced here so a single ``pytest`` run covers them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHECKS = Path(__file__).parent / "js" / "cluster_checks.js"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available to run the front-end checks")
def test_the_orb_field_obeys_its_invariants():
    """Prevents, in one run:

    * overlapping snapshot windows spawning an orb twice (the client dedupes on
      a monotonic sequence number, so a dropped frame costs nothing);
    * a backgrounded tab replaying thousands of missed pulses as a stampede;
    * budgeting the orb count against the 2px core instead of the bloom, which
      saturates the field into a wash at any real pulse rate;
    * an unjittered burst travelling as one rigid elongated smear that reads as
      a rendering artefact rather than as ten events;
    * regenerating the cached pathway sprite when the universe has not changed.
    """
    proc = subprocess.run(["node", str(CHECKS)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.stdout.strip(), f"no output from the checks: {proc.stderr[-800:]}"
    results = json.loads(proc.stdout)
    failed = [r for r in results if not r["pass"]]
    assert not failed, "\n".join(f"{r['name']}: {r['detail']}" for r in failed)
    assert len(results) >= 10
