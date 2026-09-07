"""The two telemetry streams."""

from __future__ import annotations

import pytest

from godalgo.telemetry.streams import Level, TelemetryHub


def test_a_pulse_flood_cannot_evict_readable_events():
    """Prevents: folding pulses into the event log. Pulses arrive several per
    second and events tens per hour, so a shared ring loses every readable event
    within a minute — the panel a human reads goes blank exactly when the bot
    gets busy."""
    hub = TelemetryHub(event_capacity=50, pulse_capacity=100)
    hub.event(Level.WARN, "venue", "the key was rejected")
    for i in range(5000):
        hub.pulse("BTCUSDT", "scan", f"bar {i}", 0.3)
    messages = [e["message"] for e in hub.events()]
    assert "the key was rejected" in messages


def test_both_rings_are_bounded():
    """Prevents: unbounded memory growth in a process meant to run for days."""
    hub = TelemetryHub(event_capacity=10, pulse_capacity=20)
    for i in range(500):
        hub.event(Level.INFO, "t", f"e{i}")
        hub.pulse("BTCUSDT", "scan", f"p{i}")
    assert len(hub.events(limit=1000)) == 10
    assert hub.pulse_count == 20


def test_pulse_sequence_numbers_are_monotonic_across_eviction():
    """Prevents: the client's dedupe breaking when the ring wraps. The sequence
    must keep increasing even after old pulses are evicted, or overlapping
    snapshot windows start spawning duplicate orbs."""
    hub = TelemetryHub(pulse_capacity=8)
    for i in range(100):
        hub.pulse("BTCUSDT", "scan", "r")
    window = hub.pulse_window()
    seqs = [p["seq"] for p in window]
    assert seqs == sorted(seqs)
    assert seqs[-1] == 100
    assert hub.latest_pulse_seq == 100


def test_an_unknown_pulse_kind_is_rejected_at_the_source():
    """Prevents: a typo producing orbs the front end cannot colour, which it
    then drops silently — leaving work that happened invisible."""
    hub = TelemetryHub()
    with pytest.raises(ValueError, match="unknown pulse kind"):
        hub.pulse("BTCUSDT", "decsion", "typo")


def test_the_pulse_window_is_a_window_not_the_whole_ring():
    """Prevents: sending the entire ring every second. A window with overlap is
    cheaper than tracking a cursor per socket, and the client's dedupe on seq
    makes the overlap free."""
    hub = TelemetryHub(pulse_capacity=4096)
    for i in range(1000):
        hub.pulse("BTCUSDT", "scan", "r")
    assert len(hub.pulse_window(limit=240)) == 240


def test_intensity_is_clamped():
    """Prevents: an out-of-range intensity producing an invisible or a
    blindingly opaque orb."""
    hub = TelemetryHub()
    assert hub.pulse("X", "scan", "r", 5.0).intensity == 1.0
    assert hub.pulse("X", "scan", "r", -3.0).intensity == 0.0


def test_lifetime_kind_counts_survive_ring_eviction():
    """Prevents: answering "how many bars have been evaluated" from the bounded
    ring, which silently under-reports the moment it wraps. The counts are
    lifetime totals; the ring is a recent window, and they are different
    questions."""
    hub = TelemetryHub(pulse_capacity=8)
    for _ in range(500):
        hub.pulse("BTCUSDT", "scan", "r")
    for _ in range(7):
        hub.pulse("BTCUSDT", "order", "filled")
    assert hub.pulse_count == 8              # the ring wrapped many times over
    assert hub.kind_counts["scan"] == 500    # the totals did not
    assert hub.kind_counts["order"] == 7
