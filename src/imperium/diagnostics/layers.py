"""Layered network diagnostics.

"Could not reach the venue" is the same sentence for a DNS failure, a corporate
proxy, TLS interception, a geo-block and a firewall. Five problems, five
different remedies, one useless message.

This walks the layers in order and reports the **first** one that fails, then
interprets the *pattern* of results -- which is where the answer actually is:

* fails at 4 but passes at 5 → a proxy is required, and it is working
* fails at 4 *and* 5 → the venue is blocked for this machine, and no change to
  this program will help
* passes everything but 6 → the fault is in this program, not the network, and
  it says so plainly

Two traps, both encoded:

* **Short-circuit.** Testing HTTPS against a host that does not resolve reports
  the wrong fault. Once a layer fails, the layers above it are not run; they are
  reported as "not attempted" rather than as failures.
* **A 403 block page is not "reachable".** An intercepting proxy answers with
  its own 403, and treating any HTTP reply as success puts a green tick beside
  the exact thing that is blocking you.
"""

from __future__ import annotations

import asyncio
import platform
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx

from imperium.logging_setup import safe_env_names

#: Names only. A proxy URL embeds credentials and this output is designed to be
#: pasted into a bug report.
PROXY_ENV_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
)


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIPPED = "not attempted"


@dataclass
class LayerResult:
    index: int
    name: str
    status: Status
    detail: str
    remedy: str = ""
    elapsed_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (Status.PASS, Status.WARN)


@dataclass
class Diagnosis:
    host: str
    layers: list[LayerResult] = field(default_factory=list)
    verdict: str = ""
    remedy: str = ""
    reachable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "reachable": self.reachable,
            "verdict": self.verdict,
            "remedy": self.remedy,
            "layers": [
                {
                    "index": r.index, "name": r.name, "status": r.status.value,
                    "detail": r.detail, "remedy": r.remedy,
                    "elapsed_ms": round(r.elapsed_ms, 1), **({"data": r.data} if r.data else {}),
                }
                for r in self.layers
            ],
        }

    def as_text(self) -> str:
        """Plain text, because this gets pasted into bug reports.

        Exposed at a URL as well as a button: a button in a scrolling panel is a
        control people cannot find, and a URL survives any layout change and
        needs no screenshot.
        """
        glyph = {Status.PASS: "  ok  ", Status.FAIL: " FAIL ",
                 Status.WARN: " warn ", Status.SKIPPED: "  --  "}
        lines = [
            "IMPERIUM connectivity diagnosis",
            f"host: {self.host}",
            "=" * 72,
        ]
        for r in self.layers:
            lines.append(f"[{glyph[r.status]}] {r.index}. {r.name}  ({r.elapsed_ms:.0f} ms)")
            for line in r.detail.splitlines():
                lines.append(f"          {line}")
            if r.remedy:
                for line in r.remedy.splitlines():
                    lines.append(f"       -> {line}")
        lines += ["=" * 72, "VERDICT: " + self.verdict]
        if self.remedy:
            lines.append("REMEDY:  " + self.remedy)
        return "\n".join(lines) + "\n"


def _blocked_by_intermediary(status_code: int, body: str) -> str | None:
    """Return why this HTTP reply is a block page rather than a venue reply."""
    head = body.lstrip()[:400].lower()
    if head.startswith(("<!doctype", "<html", "<?xml")) or "<title>" in head:
        return (f"the reply was an HTML page (HTTP {status_code}); the venue only "
                "ever answers in JSON, so something in between answered")
    if status_code in (403, 407, 451):
        reason = {
            403: "an intermediary refused the request",
            407: "a proxy demanded authentication",
            451: "the content was blocked for legal/regional reasons",
        }[status_code]
        return f"HTTP {status_code}: {reason}"
    if status_code >= 400:
        return f"HTTP {status_code} from the endpoint"
    return None


class NetworkDiagnostic:
    """Runs the layered probe against one venue host."""

    def __init__(self, base_url: str, probe_path: str = "/api/v3/ping",
                 timeout: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.host = urlparse(self.base_url).hostname or self.base_url
        self.probe_path = probe_path
        self.timeout = timeout

    async def run(self) -> Diagnosis:
        d = Diagnosis(host=self.host)

        env = self._layer_environment()
        d.layers.append(env)

        dns = await self._layer_dns()
        d.layers.append(dns)
        if not dns.ok:
            self._skip_from(d, 3)
            return self._conclude(d)

        stdlib = await self._layer_stdlib_https()
        d.layers.append(stdlib)

        no_env = await self._layer_httpx(trust_env=False)
        d.layers.append(no_env)

        with_env = await self._layer_httpx(trust_env=True)
        d.layers.append(with_env)

        if not (no_env.ok or with_env.ok):
            # Nothing at the HTTPS layer works. Trying the venue client next
            # would only report the same failure a second time, with a longer
            # traceback.
            self._skip_from(d, 6)
            return self._conclude(d)

        d.layers.append(await self._layer_venue_client(trust_env=with_env.ok))
        return self._conclude(d)

    @staticmethod
    def _skip_from(d: Diagnosis, start: int) -> None:
        names = {
            3: "HTTPS via the Python standard library",
            4: "HTTPS via the HTTP client, ignoring proxy environment",
            5: "HTTPS via the HTTP client, honouring proxy environment",
            6: "the venue client against a public endpoint",
        }
        for i in range(start, 7):
            d.layers.append(LayerResult(
                i, names[i], Status.SKIPPED,
                "not attempted, because an earlier layer failed and testing this "
                "one would report the wrong fault",
            ))

    # -- layers ----------------------------------------------------------

    def _layer_environment(self) -> LayerResult:
        t0 = time.perf_counter()
        set_names = safe_env_names(PROXY_ENV_NAMES)
        detail = [
            f"python {platform.python_version()} on {platform.system()} "
            f"{platform.release()} ({platform.machine()})",
            f"openssl {ssl.OPENSSL_VERSION}",
        ]
        if set_names:
            # Names only, never values.
            detail.append("proxy-related environment variables SET (names only, "
                          "values are never printed): " + ", ".join(set_names))
        else:
            detail.append("no proxy-related environment variables are set")
        status = Status.PASS
        remedy = ""
        if sys.version_info < (3, 11):
            status = Status.WARN
            remedy = "Python 3.11 or newer is required."
        return LayerResult(1, "environment", status, "\n".join(detail), remedy,
                           (time.perf_counter() - t0) * 1000,
                           data={"proxy_env_set": set_names})

    async def _layer_dns(self) -> LayerResult:
        t0 = time.perf_counter()
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                self.host, 443, proto=socket.IPPROTO_TCP
            )
        except socket.gaierror as exc:
            return LayerResult(
                2, f"DNS resolution of {self.host}", Status.FAIL,
                f"the hostname does not resolve: {exc.strerror or exc}",
                remedy=("Either this machine has no working DNS, or DNS is being "
                        "filtered to block this host. Try resolving it manually; "
                        "if other hosts resolve and this one does not, it is "
                        "being blocked deliberately."),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        addrs = sorted({i[4][0] for i in infos})
        return LayerResult(
            2, f"DNS resolution of {self.host}", Status.PASS,
            f"resolves to {', '.join(addrs[:4])}"
            + (f" (+{len(addrs) - 4} more)" if len(addrs) > 4 else ""),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            data={"addresses": addrs},
        )

    async def _layer_stdlib_https(self) -> LayerResult:
        """A TLS handshake with no third-party library involved.

        Separating this from the httpx layers is what distinguishes a broken
        certificate store or a TLS-intercepting middlebox from an httpx or proxy
        configuration problem.
        """
        t0 = time.perf_counter()
        url = f"{self.base_url}{self.probe_path}"

        def _fetch() -> tuple[int, str]:
            req = urllib.request.Request(url, headers={"User-Agent": "imperium-diagnostic"})
            # No proxy handler at all, so this is a direct connection.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=self.timeout) as resp:
                return resp.status, resp.read(2048).decode("utf-8", "replace")

        try:
            status_code, body = await asyncio.to_thread(_fetch)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(2048).decode("utf-8", "replace")
            except Exception:
                pass
            blocked = _blocked_by_intermediary(exc.code, body)
            return LayerResult(
                3, "HTTPS via the Python standard library", Status.FAIL,
                blocked or f"HTTP {exc.code} {exc.reason}",
                remedy=("An HTTP error at this layer is an answer from something, "
                        "but not necessarily from the venue."),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except urllib.error.URLError as exc:
            reason = exc.reason
            is_tls = isinstance(reason, ssl.SSLError) or "CERTIFICATE" in str(reason).upper()
            return LayerResult(
                3, "HTTPS via the Python standard library", Status.FAIL,
                f"{'TLS handshake failed' if is_tls else 'connection failed'}: {reason}",
                remedy=("A certificate verification failure here usually means a "
                        "TLS-intercepting proxy is re-signing traffic with a "
                        "private CA. That CA must be installed in the system trust "
                        "store; this program will not disable verification."
                        if is_tls else
                        "The host resolves but will not accept a direct connection: "
                        "a firewall is dropping it, or a proxy is mandatory here."),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return LayerResult(
                3, "HTTPS via the Python standard library", Status.FAIL,
                f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        blocked = _blocked_by_intermediary(status_code, body)
        if blocked:
            return LayerResult(
                3, "HTTPS via the Python standard library", Status.FAIL, blocked,
                remedy="Something answered, but it was not the venue.",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        return LayerResult(
            3, "HTTPS via the Python standard library", Status.PASS,
            f"HTTP {status_code}, JSON body ({len(body)} bytes) — a direct "
            "connection works",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    async def _layer_httpx(self, *, trust_env: bool) -> LayerResult:
        index = 5 if trust_env else 4
        name = ("HTTPS via the HTTP client, honouring proxy environment"
                if trust_env else
                "HTTPS via the HTTP client, ignoring proxy environment")
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=trust_env,
                                         follow_redirects=False) as client:
                resp = await client.get(f"{self.base_url}{self.probe_path}")
        except httpx.ProxyError as exc:
            return LayerResult(
                index, name, Status.FAIL,
                f"the configured proxy refused the request: {exc}",
                remedy=("A proxy is set in the environment and it rejected the "
                        "connection to this host. That is a policy decision at the "
                        "proxy, not a fault in this program."),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except httpx.ConnectError as exc:
            text = str(exc)
            tls = "CERTIFICATE" in text.upper() or "SSL" in text.upper()
            return LayerResult(
                index, name, Status.FAIL,
                f"{'TLS verification failed' if tls else 'connection failed'}: {text}",
                remedy=("A private CA is intercepting TLS. Install its certificate "
                        "in the system trust store — verification is never disabled."
                        if tls else ""),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except httpx.HTTPError as exc:
            return LayerResult(
                index, name, Status.FAIL, f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        blocked = _blocked_by_intermediary(resp.status_code, resp.text)
        if blocked:
            return LayerResult(
                index, name, Status.FAIL, blocked,
                remedy=("An HTTP reply is not the same as reaching the venue. "
                        "Something in between answered instead."),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        return LayerResult(
            index, name, Status.PASS,
            f"HTTP {resp.status_code}, JSON body ({len(resp.text)} bytes)",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    async def _layer_venue_client(self, *, trust_env: bool) -> LayerResult:
        """The last layer: our own client, against a public endpoint.

        If every layer below passed and this fails, the fault is in this
        program, and the diagnosis says so rather than sending the operator back
        to their network team.
        """
        from imperium.venues.alpaca.client import AlpacaClient, VenueError

        t0 = time.perf_counter()
        client = AlpacaClient(base_url=self.base_url, timeout=self.timeout,
                              trust_env=trust_env, max_retries=0)
        try:
            # An unauthenticated call that still proves the venue answered:
            # 401 from the venue is a *reachable* venue, which is exactly what
            # this layer tests. Anything else is a transport problem.
            try:
                await client._request("GET", "/v2/clock", needs_auth=False)
            except VenueError as exc:
                if exc.status != 401:
                    raise
            offset = 0
        except VenueError as exc:
            return LayerResult(
                6, "the venue client against a public endpoint", Status.FAIL,
                exc.message,
                remedy=exc.remedy,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return LayerResult(
                6, "the venue client against a public endpoint", Status.FAIL,
                f"{type(exc).__name__}: {exc}",
                remedy="This is a bug in this program.",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        finally:
            await client.aclose()

        return LayerResult(
            6, "the venue client against a public endpoint", Status.PASS,
            "the venue answered — reaching it is not the same as being "
            "authorised, which the Connections panel checks separately",
            "", (time.perf_counter() - t0) * 1000)

    # -- interpretation --------------------------------------------------

    def _conclude(self, d: Diagnosis) -> Diagnosis:
        by_index = {r.index: r for r in d.layers}
        dns, stdlib = by_index.get(2), by_index.get(3)
        no_env, with_env = by_index.get(4), by_index.get(5)
        venue = by_index.get(6)

        if dns and not dns.ok:
            d.verdict = f"DNS does not resolve {self.host}."
            d.remedy = dns.remedy
            return d

        direct_ok = bool((stdlib and stdlib.ok) or (no_env and no_env.ok))
        proxied_ok = bool(with_env and with_env.ok)

        if not direct_ok and not proxied_ok:
            tls_signature = any(
                r and r.ok is False and "TLS" in r.detail
                for r in (stdlib, no_env, with_env)
            )
            d.verdict = (
                f"{self.host} is not reachable from this machine at all. Every "
                "HTTPS path failed, with and without the proxy environment."
            )
            d.remedy = (
                ("A TLS-intercepting proxy is re-signing traffic and its CA is not "
                 "trusted here. Install that CA in the system trust store."
                 if tls_signature else
                 "This is a network policy — a firewall, a corporate filter, or a "
                 "regional block. No change to this program will help; the venue "
                 "has to be allowed through, or the machine has to be somewhere "
                 "it is not blocked.")
            )
            return d

        if not direct_ok and proxied_ok:
            d.reachable = True
            d.verdict = ("Reachable, but only through the proxy configured in the "
                         "environment. A direct connection is refused.")
            d.remedy = ("This is expected on a corporate network and is handled: "
                        "the venue client runs with trust_env enabled, so it uses "
                        "the same proxy.")
        elif venue and not venue.ok:
            d.reachable = False
            d.verdict = ("The network is fine — every layer beneath the venue "
                         "client passed — so this is a bug in this program, not a "
                         "problem with your network or your venue account.")
            d.remedy = (venue.remedy or "") + " Please report this with the output above."
            return d
        else:
            d.reachable = True
            d.verdict = f"{self.host} is reachable and the venue client works."

        if venue and venue.status is Status.WARN:
            d.remedy = (d.remedy + " " + venue.remedy).strip()
        return d


async def diagnose(base_url: str = "https://paper-api.alpaca.markets") -> Diagnosis:
    return await NetworkDiagnostic(base_url).run()
