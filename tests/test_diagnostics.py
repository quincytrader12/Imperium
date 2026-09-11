"""The layered network diagnostic."""

from __future__ import annotations

import httpx
import pytest

from imperium.diagnostics.layers import (
    NetworkDiagnostic, Status, _blocked_by_intermediary,
)


def test_a_403_block_page_is_not_counted_as_reachable():
    """Prevents: treating any HTTP reply as success. An intercepting proxy
    answers with its own 403, and a green tick beside the exact thing blocking
    you is worse than no diagnostic at all."""
    assert _blocked_by_intermediary(403, '{"code":0}') is not None
    assert _blocked_by_intermediary(407, "") is not None
    assert _blocked_by_intermediary(451, "") is not None


def test_an_html_body_with_http_200_is_still_a_block():
    """Prevents: a captive portal that answers 200 with a login page being read
    as a working connection."""
    assert _blocked_by_intermediary(200, "<!DOCTYPE html><html>...") is not None
    assert _blocked_by_intermediary(200, '{"serverTime": 1}') is None


async def test_layers_above_a_dns_failure_are_not_attempted():
    """Prevents: reporting the wrong fault. Testing HTTPS against a host that
    does not resolve produces a connection error, and an operator shown 'TLS
    failed' will go and investigate their certificate store instead of DNS."""
    diag = NetworkDiagnostic("https://this-host-does-not-exist.invalid")
    result = await diag.run()
    dns = next(r for r in result.layers if r.index == 2)
    assert dns.status is Status.FAIL
    for r in result.layers:
        if r.index >= 3:
            assert r.status is Status.SKIPPED
    assert "DNS" in result.verdict
    assert result.reachable is False


def test_the_environment_layer_never_prints_a_proxy_value(monkeypatch):
    """Prevents: a proxy URL's embedded credentials being printed into output
    that is explicitly designed to be pasted into a bug report."""
    monkeypatch.setenv("HTTPS_PROXY", "http://carol:letmein@proxy.corp:3128")
    diag = NetworkDiagnostic("https://api.binance.com")
    layer = diag._layer_environment()
    assert "HTTPS_PROXY" in layer.detail
    assert "letmein" not in layer.detail
    assert "carol" not in layer.detail


def test_the_text_report_is_plain_and_states_a_verdict():
    """Prevents: a diagnostic whose output cannot be pasted anywhere. It is
    exposed at a URL as well as a button, because a button in a scrolling panel
    is a control people cannot find."""
    from imperium.diagnostics.layers import Diagnosis, LayerResult

    d = Diagnosis(host="api.example", verdict="it works", remedy="nothing to do")
    d.layers.append(LayerResult(1, "environment", Status.PASS, "fine"))
    text = d.as_text()
    assert "VERDICT: it works" in text
    assert "environment" in text
    assert "<" not in text
