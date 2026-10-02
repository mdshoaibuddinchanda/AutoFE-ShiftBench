"""Bounded provenance packaging and read-only verification CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.provenance import build_provenance_package, verify_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Package or verify AutoFE-ShiftBench provenance without launching benchmark work")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=Path("."))
    common.add_argument("--dataset-list", type=Path, default=Path("config/dataset_list.yaml"))
    common.add_argument("--manifest-db", type=Path)
    common.add_argument("--manifest-path", type=Path)
    common.add_argument("--ledger", type=Path)
    common.add_argument("--run-id")

    package = subparsers.add_parser("package", parents=[common], help="Write a metadata-only provenance package")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--analysis-dir", action="append", default=[])

    verify = subparsers.add_parser("verify", parents=[common], help="Read-only integrity and compatibility check")
    verify.add_argument("--package-dir", type=Path)

    args = parser.parse_args()
    if args.command == "package":
        output = build_provenance_package(
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            dataset_list_path=args.dataset_list,
            manifest_db=args.manifest_db,
            manifest_path=args.manifest_path,
            ledger_path=args.ledger,
            run_id=args.run_id,
            analysis_dirs=args.analysis_dir,
        )
        print(json.dumps({"output_dir": str(output), "status": "written"}, sort_keys=True))
        return
    result = verify_provenance(
        repo_root=args.repo_root,
        dataset_list_path=args.dataset_list,
        manifest_db=args.manifest_db,
        ledger_path=args.ledger,
        run_id=args.run_id,
        package_dir=args.package_dir,
    )
    print(json.dumps(result, sort_keys=True))
    if result["overall_status"] == "attention_required":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
