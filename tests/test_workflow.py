"""The CI workflow's own inline scripts.

These are Python programs that only ever run on a GitHub runner, so nothing else
in this suite executes them -- which is how five consecutive builds failed on a
one-line mistake nobody could see locally.

The failure was worth writing a test for. This comment was in an inline script:

    # read_text()/open() without encoding= picks up the platform default.

Python scans the first two lines of every source file for a PEP 263 declaration,
matching ``coding[:=]\\s*([-\\w.]+)``. The six letters of "en**coding**" followed
by "=" matched, so Python tried to load a codec named "picks" and died with
``SyntaxError: encoding problem: picks`` before running a line.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build.yml"

#: Python's own PEP 263 pattern, from the language reference.
PEP263 = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")


def inline_python_steps() -> list[tuple[str, str]]:
    """Return (step name, script) for every ``shell: python`` step."""
    assert yaml is not None
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []):
            if step.get("shell") == "python" and "run" in step:
                out.append((step.get("name", "unnamed step"), step["run"]))
    return out


@pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
def test_the_workflow_has_inline_python_to_check():
    """Prevents: this file silently testing nothing after a workflow rewrite."""
    assert len(inline_python_steps()) >= 2


@pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
def test_no_inline_script_accidentally_declares_a_source_encoding():
    """Prevents the exact failure that broke five builds: a comment in the first
    two lines containing "coding" followed by ':' or '=' is a PEP 263 encoding
    declaration, and Python refuses to run the file if the word after it is not
    a codec. It is invisible in review and cannot fail locally."""
    for name, script in inline_python_steps():
        for lineno, line in enumerate(script.splitlines()[:2], 1):
            match = PEP263.match(line)
            if match:
                codec = match.group(1)
                try:
                    "x".encode(codec)
                except LookupError:
                    pytest.fail(
                        f"step {name!r} line {lineno} reads as a PEP 263 encoding "
                        f"declaration for a codec named {codec!r}, which does not "
                        f"exist. Python will refuse to run it.\n  {line}"
                    )


@pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
def test_every_inline_script_actually_runs():
    """Prevents: shipping a workflow whose scripts fail on the runner for a
    reason that has nothing to do with the thing they check. They are run here
    against this repository, which is the same input they get in CI."""
    root = WORKFLOW.resolve().parents[2]
    for name, script in inline_python_steps():
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True, text=True, cwd=root, timeout=120,
            encoding="utf-8",
        )
        assert proc.returncode == 0, (
            f"the inline script for step {name!r} failed in CI conditions:\n"
            f"{(proc.stdout + proc.stderr)[-2000:]}"
        )


@pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
def test_every_file_the_workflow_runs_is_tracked_by_git():
    """Prevents: CI failing on a file that exists locally but was never
    committed.

    This happened: .gitignore carried a bare ``*.spec`` to exclude the spec
    PyInstaller generates, and it also excluded the hand-written
    ``packaging/godalgo.spec``. The local build worked because the file was on
    disk; the Windows job failed with "Spec file not found" for a file that had
    never entered the repository. Existing on disk is not evidence.
    """
    import subprocess

    root = WORKFLOW.resolve().parents[2]
    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True,
        check=True).stdout.split())

    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    referenced: set[str] = set()
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run") or ""
            # Any repo-relative path with an extension mentioned in a shell step.
            for token in re.findall(r"[\w./\\-]+\.(?:spec|py|txt|cfg|toml|json|bat|ps1)",
                                    run):
                candidate = token.replace("\\", "/").lstrip("./")
                if (root / candidate).exists():
                    referenced.add(candidate)

    assert referenced, "no repo files were found referenced by the workflow"
    missing = sorted(referenced - tracked)
    assert not missing, (
        f"the workflow runs these files but git does not track them, so they "
        f"exist only on this machine and CI cannot see them: {missing}"
    )
