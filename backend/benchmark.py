"""Local benchmark harness for TRIBE compare latency."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TRIBE compare latency on local assets.")
    parser.add_argument("asset_a", type=Path, help="First asset path.")
    parser.add_argument("asset_b", type=Path, help="Second asset path.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["fast", "full"],
        choices=["fast", "full"],
        help="Profiles to benchmark.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Requested TRIBEV2 device.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 12, 16],
        help="Fast-profile batch sizes to benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_a = args.asset_a.expanduser().resolve()
    asset_b = args.asset_b.expanduser().resolve()
    if not asset_a.is_file():
        raise FileNotFoundError(f"Asset A does not exist: {asset_a}")
    if not asset_b.is_file():
        raise FileNotFoundError(f"Asset B does not exist: {asset_b}")

    import os

    os.environ["TRIBEV2_DEVICE"] = args.device
    os.environ["TRIBE_FIXTURE_MODE"] = "0"
    os.environ["TRIBE_FIXTURE_FALLBACK"] = "0"

    from backend.app import analyze_assets_compare, clear_model_cache

    print(f"Benchmarking {asset_a.name} vs {asset_b.name} on device={args.device}")
    print("")

    for profile in args.profiles:
        batch_sizes = args.batch_sizes if profile == "fast" else [None]
        for batch_size in batch_sizes:
            os.environ["TRIBEV2_PROFILE"] = profile
            if batch_size is not None:
                os.environ["TRIBEV2_BATCH_SIZE_FAST"] = str(batch_size)

            clear_model_cache(reset_mps_fallback=True)

            cold = analyze_assets_compare(asset_a, asset_b)
            warm = analyze_assets_compare(asset_a, asset_b)
            cold_diag = cold.get("diagnostics", {})
            warm_diag = warm.get("diagnostics", {})

            title = f"profile={profile}"
            if batch_size is not None:
                title += f" batch_size={batch_size}"
            print(title)
            print(f"  cold total_ms={_fmt_ms(cold_diag.get('total_ms'))} device={cold_diag.get('device_resolved')}")
            print(
                "  cold stages "
                f"load={_fmt_ms(cold_diag.get('model_load_ms'))} "
                f"build={_fmt_ms(cold_diag.get('event_build_ms'))} "
                f"predict={_fmt_ms(cold_diag.get('predict_ms'))} "
                f"summarize={_fmt_ms(cold_diag.get('summarize_ms'))}"
            )
            print(f"  warm total_ms={_fmt_ms(warm_diag.get('total_ms'))} device={warm_diag.get('device_resolved')}")
            print(
                "  warm stages "
                f"load={_fmt_ms(warm_diag.get('model_load_ms'))} "
                f"build={_fmt_ms(warm_diag.get('event_build_ms'))} "
                f"predict={_fmt_ms(warm_diag.get('predict_ms'))} "
                f"summarize={_fmt_ms(warm_diag.get('summarize_ms'))}"
            )
            print("")

    return 0


def _fmt_ms(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
