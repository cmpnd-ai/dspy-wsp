"""Command line: inspect and apply patches, and check decision models.

    dspy-wsp show  PATCH_DIR
    dspy-wsp apply BASE_MODEL_DIR PATCH_DIR OUT_DIR
    dspy-wsp check MODEL [--revision REV] [--framework mlx|torch] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from dspy_wsp.patch import Patch, PatchError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dspy-wsp", description="Inspect and apply weight-space patches.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="summarize a patch")
    s.add_argument("patch")
    a = sub.add_parser("apply", help="write a full checkpoint: base model + patch, with hash checks")
    a.add_argument("base")
    a.add_argument("patch")
    a.add_argument("out")
    c = sub.add_parser("check", help="check that a decision model loads, answers, batches and takes edits")
    from dspy_wsp.decision.families import FAMILIES

    aliases = ", ".join(f'"{a}"' for f in FAMILIES for a in f.aliases)
    c.add_argument("model", help=f"an alias ({aliases}), a Hugging Face repo id or a local folder")
    c.add_argument("--revision")
    c.add_argument("--framework", choices=["mlx", "torch"])
    c.add_argument("--json", action="store_true", help="print the report as JSON")
    args = p.parse_args(argv)
    if args.cmd == "check":
        from dspy_wsp.decision.loading import check_decision_model, report_dict

        report = check_decision_model(args.model, revision=args.revision, framework=args.framework)
        print(json.dumps(report_dict(report), indent=1) if args.json else report.summary())
        return 0 if report.ok else 1
    patch = Patch.load(args.patch)
    if args.cmd == "show":
        print(patch.summary())
        if patch.meta:
            print(json.dumps(patch.meta, indent=1, default=str))
        return 0
    try:
        out = patch.write_checkpoint(args.base, args.out)
    except PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({len(patch)} tensors edited, hashes verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
