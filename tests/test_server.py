"""The server's security properties and its contract with the UI."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from godalgo.server.app import create_app, validate_bind_host
from godalgo.session import TradingSession

KEY = "PK" + "A" * 62
SECRET = "S3cr3t" + "x" * 58


@pytest.fixture
def app_client():
    with TestClient(create_app(TradingSession())) as c:
        yield c


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "10.0.0.1", "::"])
def test_the_server_refuses_to_bind_anywhere_but_loopback(host):
    """Prevents: publishing an unauthenticated trading control plane to the local
    network. This process holds API keys and has no authentication of its own,
    so the bind address is a rule, not a default."""
    with pytest.raises(ValueError, match="refusing to bind"):
        validate_bind_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_binds_are_permitted(host):
    """Prevents: over-tightening the rule above into something unusable."""
    assert validate_bind_host(host) == host


def test_no_endpoint_ever_returns_a_credential(app_client):
    """Prevents: a secret leaving the process over HTTP. Not once, not for
    debugging. Every response body is searched for the actual secret."""
    app_client.post("/api/connections", json={
        "name": "main", "venue": "binance_spot", "api_key": KEY, "secret": SECRET})
    for url in ("/api/connections", "/api/snapshot", "/api/health"):
        body = app_client.get(url).text
        assert SECRET not in body, f"{url} leaked the secret"
        assert SECRET[:16] not in body, f"{url} leaked part of the secret"
        assert KEY not in body, f"{url} leaked the API key"


def test_a_stored_key_arrives_not_tradeable(app_client):
    """Prevents: a form POST arming real money. Storing a key and authorising it
    are two decisions."""
    r = app_client.post("/api/connections", json={
        "name": "main", "venue": "binance_spot", "api_key": KEY, "secret": SECRET})
    assert r.status_code == 201
    assert r.json()["trade_enabled"] is False


def test_going_live_needs_the_typed_phrase(app_client):
    """Prevents: a one-click button that arms real money. Two independent gates:
    the key must be marked tradeable AND the phrase must be typed exactly."""
    app_client.post("/api/connections", json={
        "name": "main", "venue": "binance_spot", "api_key": KEY, "secret": SECRET})
    app_client.post("/api/connections/main/trade", json={"trade_enabled": True})

    r = app_client.post("/api/session/mode", json={"mode": "live"})
    assert r.status_code in (403, 502)
    r = app_client.post("/api/session/mode",
                        json={"mode": "live", "confirmation": "go live"})
    assert r.status_code in (403, 502)
    assert "live" not in app_client.get("/api/snapshot").json()["mode"]


def test_the_ui_has_no_endpoint_that_places_an_order(app_client):
    """Prevents: a second, untested path to the exchange. The UI starts and stops
    sessions and switches mode; it never submits an order."""
    paths = {r.path for r in app_client.app.routes if hasattr(r, "path")}
    for path in paths:
        assert "order" not in path.lower(), f"{path} exposes order submission"
        assert "trade/execute" not in path.lower()


def test_diagnose_is_available_as_plain_text_at_a_url(app_client):
    """Prevents: a diagnostic reachable only through a button in a scrolling
    panel — a control people cannot find, whose output needs a screenshot."""
    r = app_client.get("/diagnose")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "VERDICT:" in r.text
    assert "<html" not in r.text.lower()


def test_the_diagnostic_output_contains_no_proxy_values(app_client, monkeypatch):
    """Prevents: a proxy URL's credentials being printed into output designed to
    be pasted into a bug report."""
    monkeypatch.setenv("HTTPS_PROXY", "http://dave:hunter2@proxy.corp:3128")
    text = app_client.get("/diagnose").text
    assert "hunter2" not in text
    assert "dave" not in text


def test_the_snapshot_carries_everything_the_ui_needs(app_client):
    """Prevents: a UI that silently renders nothing because a key was renamed."""
    s = app_client.get("/api/snapshot").json()
    for key in ("watchlist", "positions", "fills", "events", "pulses",
                "pulse_seq", "health", "lamps", "mode", "equity",
                "gross_exposure", "per_symbol_budget", "halted", "feed"):
        assert key in s, f"snapshot is missing {key!r}"
    for lamp in ("link", "venue", "data", "key", "session"):
        assert lamp in s["lamps"]


def test_watchlist_rows_carry_the_four_distinct_verdicts(app_client):
    """Prevents: collapsing 'the concurrency limit is full' into 'rejected',
    which tells the operator the scanner dislikes a symbol it actually likes."""
    s = app_client.get("/api/snapshot").json()
    assert s["watchlist"], "the watchlist must exist before any key is added"
    for row in s["watchlist"]:
        assert row["verdict"] in {"trading", "not_admitted", "rejected", "unscanned"}
        assert "decision" in row


def test_the_watchlist_works_with_no_credential_at_all(app_client):
    """Prevents: requiring a key before anything renders. Live prices and the
    whole scanner must work with no credential stored."""
    s = app_client.get("/api/snapshot").json()
    assert s["credential"] is None
    assert s["lamps"]["key"] == "off"
    assert len(s["watchlist"]) > 0


def test_a_missing_credential_file_does_not_break_the_page(app_client):
    """Prevents: a first run with no ~/.godalgo failing to serve the UI."""
    assert app_client.get("/").status_code == 200
    assert app_client.get("/api/connections").json()["credentials"] == []
