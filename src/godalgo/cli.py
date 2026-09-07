"""Command line entry point.

The Phase-1 deliverable is ``godalgo balance``: read a key from the store, call
the account endpoint, print the balances or a specific, actionable reason it
could not. Everything else was built only after that worked.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from decimal import Decimal
from typing import Any

from godalgo import config, logging_setup
from godalgo.logging_setup import mask


def _print_err(*parts: str) -> None:
    print(*parts, file=sys.stderr)


def _store():
    from godalgo.security.credentials import CredentialStore

    store = CredentialStore()
    report = store.permission_report
    if not report.ok:
        # Loud, because a world-readable credentials file is the whole game.
        _print_err("!" * 72)
        _print_err(f"! INSECURE CREDENTIAL FILE: {report.detail}")
        if report.remedy:
            _print_err(f"! Fix it with: {report.remedy}")
        _print_err("!" * 72)
    return store


async def _client_for(cred, *, timeout: float = 15.0):
    from godalgo.venues import registry
    from godalgo.venues.registry import load_client_factory

    spec = registry.get(cred.venue)
    factory = load_client_factory(spec)
    return factory(cred.api_key, cred.secret, base_url=spec.base_url, timeout=timeout)


# -- commands ------------------------------------------------------------

def cmd_keys_list(args: argparse.Namespace) -> int:
    store = _store()
    if not len(store):
        print(f"No credentials stored in {store.path}.")
        print("Add one with:  godalgo keys add --name main --key <API-KEY> --secret <SECRET>")
        return 0
    print(f"{'NAME':<16} {'VENUE':<14} {'API KEY':<14} {'TRADEABLE':<10} NOTE")
    for c in store.masked_list():
        print(f"{c['name']:<16} {c['venue']:<14} {c['api_key_masked']:<14} "
              f"{'yes' if c['trade_enabled'] else 'NO':<10} {c['note']}")
    print()
    print("Secrets are never printed, by this command or any other.")
    return 0


def cmd_keys_add(args: argparse.Namespace) -> int:
    from godalgo.security.credentials import CredentialError

    store = _store()
    key = args.key
    secret = args.secret
    if not key or not secret:
        import getpass

        key = key or getpass.getpass("API key: ").strip()
        secret = secret or getpass.getpass("API secret: ").strip()
    try:
        cred = store.add(args.name, args.venue, key, secret, note=args.note or "")
    except CredentialError as exc:
        _print_err(f"error: {exc}")
        if exc.remedy:
            _print_err(f"  {exc.remedy}")
        return 2
    print(f"Stored {cred.name!r} for {cred.venue} as {mask(cred.api_key)}.")
    print(f"File: {store.path} ({store.permission_report.detail})")
    print()
    print("This key is NOT enabled for trading. Pasting a key cannot by itself")
    print("authorise orders. To allow it to trade:")
    print(f"    godalgo keys enable {cred.name}")
    return 0


def cmd_keys_enable(args: argparse.Namespace) -> int:
    from godalgo.security.credentials import CredentialError

    store = _store()
    try:
        cred = store.set_trade_enabled(args.name, not args.disable)
    except CredentialError as exc:
        _print_err(f"error: {exc}")
        if exc.remedy:
            _print_err(f"  {exc.remedy}")
        return 2
    state = "ENABLED for trading" if cred.trade_enabled else "disabled for trading"
    print(f"{cred.name!r} is now {state}.")
    if cred.trade_enabled:
        print("Going live still requires typing the confirmation phrase in the UI.")
    return 0


def cmd_keys_remove(args: argparse.Namespace) -> int:
    store = _store()
    store.remove(args.name)
    print(f"Removed {args.name!r} (if it existed).")
    return 0


async def _balance(args: argparse.Namespace) -> int:
    from godalgo.security.credentials import CredentialError
    from godalgo.venues.binance.client import VenueError

    store = _store()
    try:
        cred = store.require(args.name)
    except CredentialError as exc:
        _print_err(f"error: {exc}")
        if exc.remedy:
            _print_err(f"  {exc.remedy}")
        return 2

    client = await _client_for(cred)
    try:
        try:
            await client.sync_time()
        except VenueError:
            # A failure here is not fatal; the account call below will produce a
            # better-targeted error, and reporting this one first would bury it.
            pass
        rows = await client.balances(hide_dust=not args.all)
    except VenueError as exc:
        _print_err("")
        _print_err(f"Could not read the account for {cred.name!r} ({mask(cred.api_key)}).")
        _print_err("")
        _print_err(f"  What happened: {exc.message}")
        if exc.code is not None:
            _print_err(f"  Venue code:    {exc.code}")
        if exc.remedy:
            _print_err(f"  What to do:    {exc.remedy}")
        _print_err("")
        _print_err("  If you suspect the network rather than the key, run:  godalgo diagnose")
        return 1
    finally:
        await client.aclose()

    if not rows:
        print("The account has no non-zero balances.")
        return 0
    print(f"Balances for {cred.name!r} ({mask(cred.api_key)}) on {cred.venue}:")
    print(f"  {'ASSET':<10} {'FREE':>20} {'LOCKED':>20}")
    for r in rows:
        print(f"  {r['asset']:<10} {str(r['free']):>20} {str(r['locked']):>20}")
    return 0


def cmd_balance(args: argparse.Namespace) -> int:
    return asyncio.run(_balance(args))


async def _diagnose(args: argparse.Namespace) -> int:
    from godalgo.diagnostics.layers import NetworkDiagnostic
    from godalgo.venues import registry

    spec = registry.get(args.venue)
    result = await NetworkDiagnostic(spec.base_url).run()
    sys.stdout.write(result.as_text())
    return 0 if result.reachable else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    return asyncio.run(_diagnose(args))


def cmd_serve(args: argparse.Namespace) -> int:
    from godalgo.server.app import run_server

    return run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from godalgo.strategy.calibration import main as calibrate_main

    return calibrate_main(trials=args.trials, seed=args.seed, window=args.window)


# -- parser --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="godalgo",
        description="Algorithmic crypto trading terminal for Binance Spot.",
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="debug logging (credentials are still never logged)")
    sub = p.add_subparsers(dest="command", required=True)

    keys = sub.add_parser("keys", help="manage stored API credentials")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)

    kl = keys_sub.add_parser("list", help="list stored keys (masked)")
    kl.set_defaults(func=cmd_keys_list)

    ka = keys_sub.add_parser("add", help="store a key (not tradeable by default)")
    ka.add_argument("--name", required=True)
    ka.add_argument("--venue", default="binance_spot")
    ka.add_argument("--key", default=None, help="omit to be prompted without echo")
    ka.add_argument("--secret", default=None, help="omit to be prompted without echo")
    ka.add_argument("--note", default="")
    ka.set_defaults(func=cmd_keys_add)

    ke = keys_sub.add_parser("enable", help="allow a specific key to place orders")
    ke.add_argument("name")
    ke.add_argument("--disable", action="store_true")
    ke.set_defaults(func=cmd_keys_enable)

    kr = keys_sub.add_parser("remove", help="delete a stored key")
    kr.add_argument("name")
    kr.set_defaults(func=cmd_keys_remove)

    b = sub.add_parser("balance", help="read real balances for a stored key")
    b.add_argument("--name", default="main")
    b.add_argument("--all", action="store_true", help="include zero balances")
    b.set_defaults(func=cmd_balance)

    d = sub.add_parser("diagnose", help="layered connectivity diagnosis")
    d.add_argument("--venue", default="binance_spot")
    d.set_defaults(func=cmd_diagnose)

    s = sub.add_parser("serve", help="run the terminal UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)

    c = sub.add_parser("calibrate", help="measure the regime classifier against nulls")
    c.add_argument("--trials", type=int, default=1000)
    c.add_argument("--seed", type=int, default=20240517)
    c.add_argument("--window", type=int, default=250)
    c.set_defaults(func=cmd_calibrate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging_setup.configure(logging.DEBUG if args.verbose else logging.INFO)
    config.ensure_home()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _print_err("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
