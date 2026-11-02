"""Command line: inspect and apply patches.

    dspy-wsp show  PATCH_DIR
    dspy-wsp apply BASE_MODEL_DIR PATCH_DIR OUT_DIR
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
    args = p.parse_args(argv)
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
