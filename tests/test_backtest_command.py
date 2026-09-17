"""The backtest, reachable from the thing people were actually given.

For most of this program's life the backtest was ``scripts/backtest_sector_trend.py``,
and the only build anyone outside this repository has is a .zip containing an
.exe. PyInstaller bundles the ``imperium`` package and nothing else, so what
shipped was a strategy with a leverage cap, a volatility target and a published
paper behind it -- and no way for its operator to measure any of that before
arming it. "Arm it and find out" is not a substitute for a backtest.

So these tests are mostly about reachability rather than arithmetic: the
numbers are covered in test_sector_backtest.py, and what is covered here is
that a person holding only the .exe can get them.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"

sys.path.insert(0, str(PACKAGING))
import launcher                                            # noqa: E402

from imperium.strategy import backtest_cli                  # noqa: E402


def _bars(folder: Path, symbols=("AAA", "BBB", "CCC"), days: int = 420) -> Path:
    """Bars that trend, so the run has something to trade.

    The same generator CI uses, kept here rather than imported from
    verify_build so this file does not depend on a script that only ever runs
    on a runner.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for offset, symbol in enumerate(symbols):
        period = 70 + 23 * offset
        price = 100.0 + 10.0 * offset
        lines = ["date,close"]
        for day in range(days):
            price *= (1.0 + 0.004 * math.sin(2 * math.pi * day / period)
                      + 0.0015 * math.sin(day * 1.7 + offset))
            stamp = (f"{2015 + day // 252:04d}-{(day % 252) // 21 + 1:02d}-"
                     f"{(day % 21) + 1:02d}")
            lines.append(f"{stamp},{price:.4f}")
        (folder / f"{symbol}.csv").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    return folder


def _env(home: Path) -> dict[str, str]:
    """A clean run that is still a runnable one.

    The obvious way to isolate this -- hand the subprocess a dict of three
    variables and nothing else -- passes on Linux and fails on Windows before
    a line of our code runs. ``_overlapped`` needs Winsock, Winsock needs
    ``SystemRoot``, and without it ``import asyncio`` dies with
    ``WinError 10106: the requested service provider could not be loaded``,
    which looks exactly like a packaging fault and is not one.

    So the environment is inherited and then narrowed: the home directory is
    redirected so the run cannot touch the operator's real ~/.imperium, and
    every IMPERIUM_ and SECTOR_ variable is dropped so a runner or a developer
    with the sleeve configured does not quietly change which universe is being
    backtested. USERPROFILE as well as HOME, because that is the one Windows
    reads.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("IMPERIUM_", "SECTOR_"))}
    env["IMPERIUM_NO_PAUSE"] = "1"
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return env


# -- where the code lives ---------------------------------------------------


def test_the_backtest_needs_nothing_from_the_scripts_folder():
    """The whole point of the move.

    ``scripts/`` is not in the bundle and never will be -- PyInstaller follows
    imports from one entry point, and a folder of standalone programs beside
    the repository is not reachable from it. Anything the shipped build must be
    able to run has to live under ``src/imperium``.
    """
    source = Path(backtest_cli.__file__).resolve()
    assert (ROOT / "src" / "imperium") in source.parents, (
        f"the backtest lives at {source}, which the build does not ship")

    launcher_source = (PACKAGING / "launcher.py").read_text(encoding="utf-8")
    assert "scripts" not in launcher_source, (
        "the packaged launcher reaches into scripts/, which is not bundled")


def test_the_checkout_script_delegates_rather_than_keeping_a_second_copy():
    """Two copies of a backtest is how you get two different answers.

    The checkout script stays, because ``uv run python scripts/...`` is in the
    README and in muscle memory. What it must not do is keep its own version of
    the run, which would drift from the packaged one and be discovered by
    someone comparing a report to a report.
    """
    script = (ROOT / "scripts" / "backtest_sector_trend.py").read_text(
        encoding="utf-8")
    assert "from imperium.strategy.backtest_cli import main" in script
    for owned in ("BacktestConfig(", "for mode in", "sanity_check"):
        assert owned not in script, (
            f"the checkout script still runs its own {owned!r}; it should call "
            f"the packaged one")


# -- the launcher's dispatch ------------------------------------------------


def test_backtest_is_recognised_as_the_first_argument():
    assert launcher.backtest_argv(["--backtest"]) == []
    assert launcher.backtest_argv(
        ["--backtest", "--start", "2010-01-01"]) == ["--start", "2010-01-01"]


def test_an_ordinary_start_is_left_alone():
    assert launcher.backtest_argv([]) is None
    assert launcher.backtest_argv(["--port", "9000"]) is None
    assert launcher.backtest_argv(["--no-browser"]) is None


def test_the_backtests_own_flags_are_not_eaten_by_the_launcher():
    """``--port`` means one thing to the launcher and nothing to the backtest.

    Everything after ``--backtest`` belongs to the backtest, including names
    the launcher also defines. The alternative -- a subparser -- would have to
    be taught every flag the backtest has, and the failure mode of forgetting
    one is the launcher silently swallowing it.
    """
    passed = launcher.backtest_argv(["--backtest", "--csv", "./bars",
                                     "--slippage-bps", "0"])
    assert passed == ["--csv", "./bars", "--slippage-bps", "0"]


def test_a_misplaced_backtest_flag_is_refused_rather_than_ignored():
    """``--port 9000 --backtest`` is not a request that means anything.

    Ignoring it would start the terminal while the operator sat watching for a
    report that was never going to come.
    """
    done = subprocess.run(
        [sys.executable, str(PACKAGING / "launcher.py"), "--port", "9000",
         "--backtest"],
        capture_output=True, text=True, timeout=120)
    assert done.returncode != 0
    assert "--backtest must come first" in done.stderr


# -- end to end -------------------------------------------------------------


def test_running_it_through_the_launcher_produces_a_report(tmp_path):
    """The actual claim: this command works.

    Run as a subprocess through ``launcher.py`` rather than by calling
    ``backtest_cli.main`` directly, because the part that was broken was never
    the arithmetic -- it was the path from the entry point people have to the
    code that does the work.
    """
    done = subprocess.run(
        [sys.executable, str(PACKAGING / "launcher.py"), "--backtest",
         "--csv", str(_bars(tmp_path / "bars"))],
        capture_output=True, text=True, timeout=600, env=_env(tmp_path),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for expected in ("near_close", "next_open", "leverage cap 1.0x",
                     "leverage cap 2.0x", "Max drawdown", "Final equity"):
        assert expected in done.stdout, f"the report has no {expected!r}"


def test_the_run_is_isolated_without_being_crippled(monkeypatch):
    """The CI failure this file caused, kept as a test.

    Handing the subprocess a hand-built dict looked like the careful thing to
    do and broke the Windows job: Python could not import asyncio at all. The
    two properties have to hold together -- our own configuration must not
    leak in, and the platform's must not be stripped out.
    """
    monkeypatch.setenv("SECTOR_TREND_UNIVERSE", "XLK,XLF")
    monkeypatch.setenv("IMPERIUM_OPERATOR", "Somebody Else")
    env = _env(Path("/tmp/home"))

    assert "SECTOR_TREND_UNIVERSE" not in env, (
        "a configured universe leaked in; the run would backtest something "
        "other than the default and nothing would say so")
    assert "IMPERIUM_OPERATOR" not in env
    assert env["IMPERIUM_NO_PAUSE"] == "1"
    assert env["HOME"] == env["USERPROFILE"] == "/tmp/home"

    # Everything the platform needs is still there. On Windows the one that
    # matters is SystemRoot; on POSIX, PATH.
    for name in ("SystemRoot", "PATH"):
        if name in os.environ:
            assert env[name] == os.environ[name], (
                f"{name} was stripped; on Windows this is the difference "
                f"between a backtest and WinError 10106")


def test_the_run_writes_its_home_where_it_was_told_to(tmp_path):
    """Isolation that is claimed and not checked is not isolation.

    If the redirect silently failed, the test would still pass -- and it would
    be creating and writing a real ~/.imperium on whoever ran it.
    """
    subprocess.run(
        [sys.executable, str(PACKAGING / "launcher.py"), "--backtest",
         "--csv", str(_bars(tmp_path / "bars"))],
        capture_output=True, text=True, timeout=600, env=_env(tmp_path),
    )
    assert (tmp_path / ".imperium").is_dir(), (
        "the run did not use the home it was given, so it used the real one")


def test_the_report_carries_its_own_warnings(tmp_path):
    """A backtest that only prints its result is an advertisement.

    These bars are smooth by construction, so a trend follower walks them with
    a Sharpe no real strategy reaches. The report has to say so on its own --
    the reader it exists for is the one who wants the number to be good.
    """
    done = subprocess.run(
        [sys.executable, str(PACKAGING / "launcher.py"), "--backtest",
         "--csv", str(_bars(tmp_path / "bars"))],
        capture_output=True, text=True, timeout=600, env=_env(tmp_path),
    )
    assert "CHECK BEFORE BELIEVING THIS" in done.stdout, (
        "a Sharpe far above the paper's drew no warning")
    assert "the paper's own 2005-2024 figures" in done.stdout.lower()


def test_it_places_no_orders_and_arms_nothing(tmp_path):
    """The one thing an operator must be able to assume before running it.

    The .bat says "nothing is ordered and nothing is armed", and the command is
    reached from a build that holds a live key. Checked at the source rather
    than by watching a venue: the backtest must not import the order path at
    all.
    """
    source = Path(backtest_cli.__file__).read_text(encoding="utf-8")
    for forbidden in ("submit_order", "place_order", "arm_by_hand",
                      "trade_enabled"):
        assert forbidden not in source, (
            f"the backtest module reaches {forbidden!r}")


# -- what ships alongside it ------------------------------------------------


def test_the_batch_file_ships_and_says_what_it_does():
    text = (PACKAGING / "Backtest-Sector-Trend.bat").read_text(encoding="utf-8")
    assert "--backtest" in text
    assert "nothing is armed" in text.lower(), (
        "the window does not tell the operator it is safe to run")


def test_the_workflow_copies_the_batch_file_into_the_zip():
    """The gap that made this whole change necessary, in its other form.

    A .bat that exists in the repository and not in the .zip is the same
    problem as a backtest that exists in scripts/ and not in the bundle.
    """
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
        encoding="utf-8")
    assert "Copy-Item packaging/Backtest-Sector-Trend.bat" in workflow


def test_ci_checks_the_frozen_build_can_actually_backtest():
    """numpy is reached from exactly one place in this program.

    An ``excludes`` entry or a hook change could drop it and every other check
    in verify_build.py would still pass -- the terminal serves its page, its
    API and its diagnostics without ever touching it.
    """
    verify = (PACKAGING / "verify_build.py").read_text(encoding="utf-8")
    assert '"--backtest"' in verify, (
        "verify_build.py never runs the backtest command")
    # The call, not the definition. A defined-but-never-called checker is the
    # shape this exact test was first written loosely enough to miss: renaming
    # it to _unused_check_backtest left the substring intact and the build
    # unchecked.
    assert "\n        check_backtest(path, failures)" in verify, (
        "check_backtest is defined but main() never calls it")


def test_cis_synthetic_bars_actually_make_the_strategy_trade(tmp_path):
    """Otherwise the CI check verifies that a report prints, and nothing else.

    A flat series takes no positions, and a backtest that takes no positions
    exercises neither the sizing, nor the leverage cap, nor the exits.
    """
    sys.path.insert(0, str(PACKAGING))
    from verify_build import synthetic_bars

    from imperium.strategy import sector_backtest as bt

    synthetic_bars(tmp_path / "bars")
    closes, dates = backtest_cli.from_csv(tmp_path / "bars")
    result = bt.run(closes, dates)
    assert len(result.trades) >= 5, (
        f"CI's bars produced {len(result.trades)} trades; the check would pass "
        f"on a build whose strategy never opened a position")


# -- the module's own edges -------------------------------------------------


def test_symbols_are_intersected_rather_than_filled():
    """A filled bar is an invented price, and a breakout rule trades it.

    A symbol halted for a week comes back with a flat line, which reads as a
    tightening range and then a breakout -- neither of which happened.
    """
    closes, dates = backtest_cli._align({
        "AAA": {"2020-01-01": 10.0, "2020-01-02": 11.0, "2020-01-03": 12.0},
        "BBB": {"2020-01-02": 20.0, "2020-01-03": 21.0},
    })
    assert dates == ["2020-01-02", "2020-01-03"]
    assert list(closes["AAA"]) == [11.0, 12.0]
    assert list(closes["BBB"]) == [20.0, 21.0]


def test_an_empty_folder_says_so_rather_than_backtesting_nothing(tmp_path):
    with pytest.raises(SystemExit) as caught:
        backtest_cli.from_csv(tmp_path)
    assert "No usable CSVs" in str(caught.value)


def test_a_csv_with_an_unreadable_row_skips_the_row_not_the_file(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "date,close\n2020-01-01,10\n2020-01-02,oops\n2020-01-03,12\n",
        encoding="utf-8")
    closes, dates = backtest_cli.from_csv(tmp_path)
    assert dates == ["2020-01-01", "2020-01-03"]
    assert list(closes["AAA"]) == [10.0, 12.0]


def test_both_execution_assumptions_are_always_reported():
    """Whether the fill is at today's close or tomorrow's open moves the result
    by more than most parameter choices do. A reader shown only the favourable
    one has been shown a number, not a measurement."""
    assert set(backtest_cli.EXEC_MODES) == {"near_close", "next_open"}
    assert 1.0 in backtest_cli.LEVERAGE_CAPS
