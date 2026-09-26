"""The settings file: where an operator actually changes anything.

This exists because of a question I could not answer. Having told the operator
several times to "set SECTOR_TREND_ENABLED=true", they asked where — and the
honest answer was nowhere. The program ships as a Windows executable launched
by double-clicking a batch file. There is no shell to export a variable in,
and telling someone to edit a .bat is not a configuration system.

So: a file beside their credentials, created on first run with every option
listed and commented out, read at startup before anything consults a setting.
"""

from __future__ import annotations

import os

import pytest

from imperium import config


def test_the_file_is_created_on_first_run_so_it_can_be_found():
    """An empty directory is not an answer to "where do I set this". The file
    exists before anyone looks for it."""
    assert config.load_settings() == ([], [])
    path = config.settings_path()
    assert path.exists()
    assert path.name == "settings.txt"
    assert path.parent == config.home_dir()


def test_the_created_file_explains_itself():
    """A blank file teaches nothing. Every option appears, commented out, with
    its default and the reason it might matter."""
    config.load_settings()
    text = config.settings_path().read_text(encoding=config.TEXT_ENCODING)
    for name in sorted(config.SETTABLE):
        assert name in text, f"{name} is not mentioned in the template"
    assert "restart IMPERIUM" in text
    # The small-account trap is the one that would otherwise be discovered by
    # watching a run place nothing.
    assert "$1.00" in text


def test_nothing_in_the_template_is_active():
    """The file must describe the defaults, not change them. A freshly created
    settings file that switched something on would arm a strategy nobody
    asked for."""
    config.load_settings()
    text = config.settings_path().read_text(encoding=config.TEXT_ENCODING)
    found, refused = config.parse_settings(text)
    assert found == {}, f"the template activates {sorted(found)}"
    assert refused == []


def test_a_setting_written_in_the_file_reaches_the_strategy():
    """The whole point, end to end."""
    from imperium.strategy.sector_config import from_environment

    config.ensure_home()
    config.settings_path().write_text(
        "# a comment\nSECTOR_TREND_ENABLED=true\nSECTOR_TREND_ALLOCATION=0.36\n",
        encoding=config.TEXT_ENCODING)
    applied, refused = config.load_settings()

    assert applied == ["SECTOR_TREND_ALLOCATION", "SECTOR_TREND_ENABLED"]
    cfg = from_environment()
    assert cfg.enabled is True
    assert cfg.allocation == pytest.approx(0.36)


def test_a_real_environment_variable_wins(monkeypatch):
    """Someone who exported a variable in a shell meant it. A file quietly
    overriding them is the kind of surprise that costs an hour to find."""
    monkeypatch.setenv("SECTOR_TREND_ALLOCATION", "0.99")
    config.ensure_home()
    config.settings_path().write_text("SECTOR_TREND_ALLOCATION=0.10\n",
                                      encoding=config.TEXT_ENCODING)
    applied, _ = config.load_settings()
    assert "SECTOR_TREND_ALLOCATION" not in applied
    assert os.environ["SECTOR_TREND_ALLOCATION"] == "0.99"


# -- what the file may not do --------------------------------------------

@pytest.mark.parametrize("name", [
    "PATH", "HTTPS_PROXY", "http_proxy", "IMPERIUM_HOME",
    "PYTHONPATH", "LD_PRELOAD", "ALPACA_API_KEY",
])
def test_the_file_cannot_set_anything_outside_its_allow_list(name):
    """THE security property here.

    This file is read straight into the process environment. Without an
    allow-list, a line in it could redirect PATH, point the program at a proxy,
    move the credential directory, or preload a library — and it is a plain
    text file in the operator's home directory, which is a much softer target
    than the code. Names are matched against a fixed set and everything else is
    refused, whatever it looks like.
    """
    config.ensure_home()
    config.settings_path().write_text(f"{name}=hostile\n",
                                      encoding=config.TEXT_ENCODING)
    before = os.environ.get(name)
    applied, refused = config.load_settings()

    assert applied == []
    assert name.upper() in refused
    assert os.environ.get(name) == before, f"{name} was overwritten"


def test_a_refused_name_is_reported_rather_than_dropped():
    """A setting that does nothing and says nothing is the worst of the three
    possible outcomes — the operator believes it took effect."""
    config.ensure_home()
    config.settings_path().write_text("SECTOR_TREND_ENABLD=true\n",
                                      encoding=config.TEXT_ENCODING)
    applied, refused = config.load_settings()
    assert applied == []
    assert refused == ["SECTOR_TREND_ENABLD"]


def test_only_names_are_returned_never_values():
    """The return value goes straight into a log line, and this file sits
    beside the credentials. Nothing read out of it is ever logged."""
    config.ensure_home()
    config.settings_path().write_text("SECTOR_TREND_ALLOCATION=0.42\n",
                                      encoding=config.TEXT_ENCODING)
    applied, refused = config.load_settings()
    assert applied == ["SECTOR_TREND_ALLOCATION"]
    assert "0.42" not in " ".join(applied + refused)


# -- robustness -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "", "   \n\n", "# only comments\n", "no equals sign here\n",
    "=novalue\n", "SECTOR_TREND_ENABLED\n",
])
def test_a_malformed_file_does_not_stop_the_terminal_starting(text):
    """Defaults are a working configuration. A typo must cost a setting, not
    a session."""
    config.ensure_home()
    config.settings_path().write_text(text, encoding=config.TEXT_ENCODING)
    applied, refused = config.load_settings()
    assert isinstance(applied, list) and isinstance(refused, list)


def test_quotes_and_spacing_are_tolerated():
    """People write settings files the way they write them."""
    found, _ = config.parse_settings(
        '  sector_trend_enabled = "true"  \nSECTOR_TREND_ALLOCATION=\'0.3\'\n')
    assert found["SECTOR_TREND_ENABLED"] == "true"
    assert found["SECTOR_TREND_ALLOCATION"] == "0.3"


def test_an_unreadable_file_is_survivable(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("nope")

    config.ensure_home()
    config.settings_path().write_text("SECTOR_TREND_ENABLED=true\n",
                                      encoding=config.TEXT_ENCODING)
    monkeypatch.setattr(type(config.settings_path()), "read_text", boom)
    assert config.load_settings() == ([], [])


def test_the_packaged_launcher_loads_settings_and_not_just_the_cli():
    """The bug this nearly shipped with.

    ``packaging/launcher.py`` is the entry the .exe actually runs; ``cli.main``
    is for a source checkout. Loading settings in the CLI and not the launcher
    would have meant the file worked perfectly for anyone running from source
    and did nothing at all for the people it was written for — who are the only
    ones who cannot set an environment variable in the first place.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    launcher = (root / "packaging" / "launcher.py").read_text(
        encoding=config.TEXT_ENCODING)
    cli = (root / "src" / "imperium" / "cli.py").read_text(
        encoding=config.TEXT_ENCODING)

    assert "load_settings()" in launcher, (
        "the packaged launcher never loads the settings file")
    assert "load_settings()" in cli, (
        "the source entry point never loads the settings file")
    # Before the server starts: strategies read their configuration once, when
    # the session is constructed, so a load afterwards would appear to do
    # nothing — the hardest kind of configuration bug to see.
    assert launcher.index("load_settings()") < launcher.index("run_server("), (
        "settings are loaded after the server starts, which is too late")


def test_the_launcher_windows_name_the_settings_file():
    """Where an operator actually looks. The batch window is the one piece of
    text they reliably read, and "where do I set this" was a question the
    program had no answer to."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "packaging"
    for name in ("Start-IMPERIUM.bat", "Run-IMPERIUM-247.bat",
                 "Backtest-Sector-Trend.bat"):
        text = (root / name).read_text(encoding=config.TEXT_ENCODING)
        assert "settings.txt" in text, f"{name} does not mention the settings file"
