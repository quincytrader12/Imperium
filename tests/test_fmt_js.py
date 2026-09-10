"""The header's clock, checked under Node.

The venue publishes its calendar in UTC. That is correct for a log and useless
on screen: an operator outside UTC converts it in their head every time they
glance at the header, and one who does not is reading a number that means
nothing to them. The countdown leads for that reason -- it is the same
sentence in every timezone -- and this checks the arithmetic behind it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

CHECKS = Path(__file__).parent / "js" / "fmt_checks.js"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available to run the front-end checks")
def test_the_header_clock_arithmetic_holds():
    # A viewer well away from UTC: the whole point is that they see their own
    # clock, and under TZ=UTC a broken conversion would look correct.
    env = dict(os.environ, TZ="Africa/Johannesburg")
    proc = subprocess.run(["node", str(CHECKS)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60, env=env)
    assert proc.stdout.strip(), f"no output: {proc.stderr[-800:]}"
    failed = [r for r in json.loads(proc.stdout) if not r["pass"]]
    assert not failed, "\n".join(f"{r['name']}: {r['detail']}" for r in failed)
