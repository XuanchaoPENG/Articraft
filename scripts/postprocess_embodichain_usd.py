from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from articraft.sdk.embodichain import export_embodichain_usdc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create an EmbodiChain-compatible USDC from an Articraft USD stage."
    )
    parser.add_argument("source", type=Path, help="Source USD, USDA, or USDC file.")
    parser.add_argument("destination", type=Path, help="Destination .usdc file.")
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy files next to the source layer for relative texture references.",
    )
    args = parser.parse_args(argv)
    print(
        export_embodichain_usdc(
            args.source,
            args.destination,
            copy_assets=args.copy_assets,
        )
    )


if __name__ == "__main__":
    main()
